// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <string>

#include "PlatformUtils.hpp"

TEST(TestPlatformUtils, GetSystemInfoReturnsNonEmpty)
{
    auto result = hipdnn_backend::platform_utilities::getSystemInfo();

    EXPECT_FALSE(result.empty());
}

TEST(TestPlatformUtils, GetSystemInfoContainsSystemName)
{
    auto result = hipdnn_backend::platform_utilities::getSystemInfo();

    EXPECT_NE(result.find("System Name:"), std::string::npos);
}

TEST(TestPlatformUtils, GetSystemInfoContainsMachine)
{
    auto result = hipdnn_backend::platform_utilities::getSystemInfo();

    EXPECT_NE(result.find("Machine:"), std::string::npos);
}
