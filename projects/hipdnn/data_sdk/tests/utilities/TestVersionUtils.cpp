// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>

using namespace hipdnn_data_sdk::utilities;

TEST(TestVersionUtils, VersionStringConstructorValidInput)
{
    Version version;
    std::vector<std::string> validVersions{"1.12.53", "1.12.53.aw939d94"};

    for(const auto& versionStr : validVersions)
    {
        ASSERT_NO_THROW(version = Version{versionStr});
        EXPECT_EQ(version, Version(1, 12, 53)) << "Version string = " << versionStr;
    }
}

TEST(TestVersionUtils, VersionTupleInvalidInput)
{
    Version version;
    std::vector<std::string> invalidVersions{"1.12", "1.12.a53", "1", "", "Str1.12.53"};

    for(const auto& versionStr : invalidVersions)
    {
        EXPECT_THROW(version = Version(versionStr), std::invalid_argument)
            << "Version string = " << versionStr;
    }
}
