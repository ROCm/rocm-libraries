// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <gtest/gtest.h>
#include <string>

#include "HipdnnException.hpp"
#include "PlatformUtils.hpp"
#include "TestPluginConstants.hpp"

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

TEST(TestPlatformUtils, GetCurrentModuleDirectoryReturnsExistingDirectory)
{
    auto result = hipdnn_backend::platform_utilities::getCurrentModuleDirectory();

    EXPECT_FALSE(result.empty());
    EXPECT_TRUE(result.is_absolute());
    EXPECT_TRUE(std::filesystem::exists(result));
    EXPECT_TRUE(std::filesystem::is_directory(result));
}

#if defined(__linux__)
namespace
{
const auto TEST_PLUGIN_DIR
    = std::filesystem::path(hipdnn_backend::plugin_constants::getTestPluginDefaultDir());
const auto TEST_PLUGIN_PATH
    = (hipdnn_backend::platform_utilities::getCurrentModuleDirectory().parent_path()
       / TEST_PLUGIN_DIR / hipdnn_data_sdk::utilities::getLibraryName(TEST_PLUGIN1_NAME));
} // namespace

TEST(TestPlatformUtils, OpenLibraryLoadsPluginAndGetsSymbol)
{
    auto handle = hipdnn_backend::platform_utilities::openLibrary(TEST_PLUGIN_PATH);
    ASSERT_NE(handle, nullptr);

    EXPECT_NE(hipdnn_backend::platform_utilities::getSymbol(handle, "hipdnnPluginGetName"),
              nullptr);

    hipdnn_backend::platform_utilities::closeLibrary(handle);
}

TEST(TestPlatformUtils, OpenLibraryThrowsHipdnnExceptionForMissingLibrary)
{
    EXPECT_THROW(
        hipdnn_backend::platform_utilities::openLibrary("libhipdnn_missing_test_library.so"),
        hipdnn_backend::HipdnnException);
}
#endif
