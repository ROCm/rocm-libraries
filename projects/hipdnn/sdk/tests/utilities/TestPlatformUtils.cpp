// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <iostream>

// TODO: More tests for PlatformUtils since it is untested

TEST(TestPlatformUtils, PathCompEqIdenticalPaths)
{
    std::filesystem::path path1 = "/home/user/project";
    std::filesystem::path path2 = "/home/user/project";

    EXPECT_TRUE(hipdnn_sdk::utilities::pathCompEq(path1, path2));
}

TEST(TestPlatformUtils, PathCompEqDifferentPaths)
{
    std::filesystem::path path1 = "/home/user/project1";
    std::filesystem::path path2 = "/home/user/project2";

    EXPECT_FALSE(hipdnn_sdk::utilities::pathCompEq(path1, path2));
}

TEST(TestPlatformUtils, PathCompEqEmptyPaths)
{
    std::filesystem::path path1;
    std::filesystem::path path2;

    EXPECT_TRUE(hipdnn_sdk::utilities::pathCompEq(path1, path2));
}
