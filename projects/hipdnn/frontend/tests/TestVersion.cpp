// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_frontend/version.h>

#include <gtest/gtest.h>

using namespace hipdnn_data_sdk::utilities;

TEST(TestVersion, ParsedSuccessfully)
{
    EXPECT_NO_THROW(Version(std::string_view{HIPDNN_FRONTEND_VERSION_STRING}));
}

TEST(TestVersion, PositiveVersion)
{
    Version version;
    ASSERT_NO_THROW(version = Version(std::string_view{HIPDNN_FRONTEND_VERSION_STRING}));

    EXPECT_GE(version.major, 0);
    EXPECT_GE(version.minor, 0);
    EXPECT_GE(version.patch, 0);
}

TEST(TestVersion, MacrosMatchVersionString)
{
    Version version;
    ASSERT_NO_THROW(version = Version(std::string_view{HIPDNN_FRONTEND_VERSION_STRING}));

    EXPECT_EQ(version.major, HIPDNN_FRONTEND_VERSION_MAJOR);
    EXPECT_EQ(version.minor, HIPDNN_FRONTEND_VERSION_MINOR);
    EXPECT_EQ(version.patch, HIPDNN_FRONTEND_VERSION_PATCH);
}

TEST(TestVersion, TweakNonEmpty)
{
    EXPECT_FALSE(std::string_view{HIPDNN_FRONTEND_VERSION_TWEAK}.empty());
}
