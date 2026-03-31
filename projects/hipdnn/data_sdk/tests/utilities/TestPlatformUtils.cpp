// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <iostream>

#if defined(__linux__)
#include <array>
#include <climits>
#include <unistd.h>
#endif

TEST(TestPlatformUtils, PathCompEqIdenticalPaths)
{
    const std::filesystem::path path1 = "/home/user/project";
    const std::filesystem::path path2 = "/home/user/project";

    EXPECT_TRUE(hipdnn_data_sdk::utilities::pathCompEq(path1, path2));
}

TEST(TestPlatformUtils, PathCompEqDifferentPaths)
{
    const std::filesystem::path path1 = "/home/user/project1";
    const std::filesystem::path path2 = "/home/user/project2";

    EXPECT_FALSE(hipdnn_data_sdk::utilities::pathCompEq(path1, path2));
}

TEST(TestPlatformUtils, PathCompEqEmptyPaths)
{
    const std::filesystem::path path1;
    const std::filesystem::path path2;

    EXPECT_TRUE(hipdnn_data_sdk::utilities::pathCompEq(path1, path2));
}

TEST(TestPlatformUtils, GetCurrentExecutableDirectoryReturnsValidPath)
{
    auto execDir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory();

    EXPECT_FALSE(execDir.empty());
}

TEST(TestPlatformUtils, GetCurrentExecutableDirectoryExists)
{
    auto execDir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory();

    EXPECT_TRUE(std::filesystem::exists(execDir));
}

TEST(TestPlatformUtils, GetCurrentExecutableDirectoryIsAbsolute)
{
    auto execDir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory();

    EXPECT_TRUE(execDir.is_absolute());
}

TEST(TestPlatformUtils, GetCurrentExecutableDirectoryIsDirectory)
{
    auto execDir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory();

    EXPECT_TRUE(std::filesystem::is_directory(execDir));
}

#if defined(__linux__)
TEST(TestPlatformUtils, GetCurrentExecutableDirectoryContainsExecutable)
{
    auto execDir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory();

    std::array<char, PATH_MAX> execPath{};
    const ssize_t len = readlink("/proc/self/exe", execPath.data(), PATH_MAX);
    ASSERT_NE(len, -1);

    const std::filesystem::path actualExecPath(
        std::string(execPath.data(), static_cast<size_t>(len)));

    EXPECT_TRUE(std::filesystem::exists(execDir / actualExecPath.filename()));
}
#endif // defined(__linux__)

// getLibraryName tests

TEST(TestPlatformUtils, GetLibraryNameBasicName)
{
    auto result = hipdnn_data_sdk::utilities::getLibraryName("foo");

#if defined(__linux__)
    EXPECT_EQ(result, "libfoo.so");
#elif defined(_WIN32)
    EXPECT_EQ(result, "foo.dll");
#endif
}

TEST(TestPlatformUtils, GetLibraryNameEmptyName)
{
    auto result = hipdnn_data_sdk::utilities::getLibraryName("");

#if defined(__linux__)
    EXPECT_EQ(result, "lib.so");
#elif defined(_WIN32)
    EXPECT_EQ(result, ".dll");
#endif
}

TEST(TestPlatformUtils, GetLibraryNameWithDots)
{
    auto result = hipdnn_data_sdk::utilities::getLibraryName("foo.bar");

#if defined(__linux__)
    EXPECT_EQ(result, "libfoo.bar.so");
#elif defined(_WIN32)
    EXPECT_EQ(result, "foo.bar.dll");
#endif
}

// getExecutableName tests

TEST(TestPlatformUtils, GetExecutableNameBasicName)
{
    auto result = hipdnn_data_sdk::utilities::getExecutableName("app");

#if defined(__linux__)
    EXPECT_EQ(result, "app");
#elif defined(_WIN32)
    EXPECT_EQ(result, "app.exe");
#endif
}

TEST(TestPlatformUtils, GetExecutableNameEmptyName)
{
    auto result = hipdnn_data_sdk::utilities::getExecutableName("");

#if defined(__linux__)
    EXPECT_EQ(result, "");
#elif defined(_WIN32)
    EXPECT_EQ(result, ".exe");
#endif
}

// getEnv tests

TEST(TestPlatformUtils, GetEnvReturnsValue)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter(
        "HIPDNN_TEST_PLATFORMUTILS_GETENV", "test_value");

    auto result = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_GETENV");

    EXPECT_EQ(result, "test_value");
}

TEST(TestPlatformUtils, GetEnvReturnsDefaultWhenUnset)
{
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_UNSET");

    auto result
        = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_UNSET", "default_value");

    EXPECT_EQ(result, "default_value");
}

TEST(TestPlatformUtils, GetEnvReturnsEmptyWhenUnsetNoDefault)
{
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_UNSET2");

    auto result = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_UNSET2");

    EXPECT_EQ(result, "");
}

TEST(TestPlatformUtils, GetEnvReturnsEmptyStringValue)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter(
        "HIPDNN_TEST_PLATFORMUTILS_EMPTY", "");

    auto result
        = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_EMPTY", "default_value");

    EXPECT_EQ(result, "");
}

// setEnv tests

TEST(TestPlatformUtils, SetEnvSetsValue)
{
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_SET");
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_TEST_PLATFORMUTILS_SET", "new_value");

    auto result = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_SET");

    EXPECT_EQ(result, "new_value");

    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_SET");
}

TEST(TestPlatformUtils, SetEnvOverwritesExisting)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter(
        "HIPDNN_TEST_PLATFORMUTILS_OVERWRITE", "original");

    hipdnn_data_sdk::utilities::setEnv("HIPDNN_TEST_PLATFORMUTILS_OVERWRITE", "updated");

    auto result = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_OVERWRITE");

    EXPECT_EQ(result, "updated");
}

// unsetEnv tests

TEST(TestPlatformUtils, UnsetEnvRemovesVariable)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_TEST_PLATFORMUTILS_REMOVE", "to_remove");
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_REMOVE");

    auto result = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_REMOVE");

    EXPECT_EQ(result, "");
}

TEST(TestPlatformUtils, UnsetEnvNoOpOnMissing)
{
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_NONEXISTENT");

    auto result = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_NONEXISTENT");

    EXPECT_EQ(result, "");
}
