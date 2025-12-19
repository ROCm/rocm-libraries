// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <iostream>

#if defined(__linux__)
#include <array>
#include <climits>
#include <unistd.h>
#endif

// TODO: More tests for PlatformUtils since it is largely untested

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

TEST(TestPlatformUtils, GetCurrentExecutableDirectoryReturnsValidPath)
{
    auto execDir = hipdnn_sdk::utilities::getCurrentExecutableDirectory();

    EXPECT_FALSE(execDir.empty());
}

TEST(TestPlatformUtils, GetCurrentExecutableDirectoryExists)
{
    auto execDir = hipdnn_sdk::utilities::getCurrentExecutableDirectory();

    EXPECT_TRUE(std::filesystem::exists(execDir));
}

TEST(TestPlatformUtils, GetCurrentExecutableDirectoryIsAbsolute)
{
    auto execDir = hipdnn_sdk::utilities::getCurrentExecutableDirectory();

    EXPECT_TRUE(execDir.is_absolute());
}

TEST(TestPlatformUtils, GetCurrentExecutableDirectoryIsDirectory)
{
    auto execDir = hipdnn_sdk::utilities::getCurrentExecutableDirectory();

    EXPECT_TRUE(std::filesystem::is_directory(execDir));
}

#if defined(__linux__)
TEST(TestPlatformUtils, GetCurrentExecutableDirectoryContainsExecutable)
{
    auto execDir = hipdnn_sdk::utilities::getCurrentExecutableDirectory();

    std::array<char, PATH_MAX> execPath{};
    ssize_t len = readlink("/proc/self/exe", execPath.data(), PATH_MAX);
    ASSERT_NE(len, -1);

    std::filesystem::path actualExecPath(std::string(execPath.data(), static_cast<size_t>(len)));

    EXPECT_TRUE(std::filesystem::exists(execDir / actualExecPath.filename()));
}
#endif // defined(__linux__)
