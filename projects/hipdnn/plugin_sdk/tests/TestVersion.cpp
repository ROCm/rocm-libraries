// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
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

// Applicability filter prerequisites for the override-execute entry point
// (RFC 0008 §B.8). The constant-aware tests live alongside the host's
// applicability filter in
// `backend/tests/plugin/` so the plugin SDK test executable does not pull
// in a `backend/src/` header (preserves the `plugin_sdk` → `data_sdk`
// linkage boundary documented in `projects/hipdnn/CLAUDE.md`). Generic
// `Version` parsing/comparison coverage stays here.

TEST(TestVersion, BaselineLessThanOnePointOne)
{
    // Plain string-driven checks; do NOT include the backend constant
    // header from the plugin SDK test executable.
    const Version baseline{std::string_view{"1.0.0"}};
    const Version onePointOne{std::string_view{"1.1.0"}};
    EXPECT_TRUE(baseline < onePointOne);
}

TEST(TestVersion, ZeroLessThanBaseline)
{
    const Version zero{std::string_view{"0.0.0"}};
    const Version baseline{std::string_view{"1.0.0"}};
    EXPECT_TRUE(zero < baseline);
}
