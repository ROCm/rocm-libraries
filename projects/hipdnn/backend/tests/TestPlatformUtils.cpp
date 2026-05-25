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

namespace
{
const auto TEST_PLUGIN_DIR
    = std::filesystem::path(hipdnn_backend::plugin_constants::getTestPluginDefaultDir());
const auto TEST_PLUGIN_PATH
    = (hipdnn_backend::platform_utilities::getCurrentModuleDirectory().parent_path()
       / TEST_PLUGIN_DIR / hipdnn_data_sdk::utilities::getLibraryName(TEST_PLUGIN1_NAME));

class LibraryHandleGuard
{
public:
    explicit LibraryHandleGuard(hipdnn_backend::platform_utilities::PluginLibHandle handle)
        : _handle(handle)
    {
    }

    ~LibraryHandleGuard()
    {
        if(_handle != nullptr)
        {
            hipdnn_backend::platform_utilities::closeLibrary(_handle);
        }
    }

    LibraryHandleGuard(const LibraryHandleGuard&) = delete;
    LibraryHandleGuard& operator=(const LibraryHandleGuard&) = delete;

    hipdnn_backend::platform_utilities::PluginLibHandle get() const
    {
        return _handle;
    }

private:
    hipdnn_backend::platform_utilities::PluginLibHandle _handle;
};
} // namespace

TEST(TestPlatformUtils, OpenLibraryLoadsPluginAndGetsSymbol)
{
    const LibraryHandleGuard library(
        hipdnn_backend::platform_utilities::openLibrary(TEST_PLUGIN_PATH));
    ASSERT_NE(library.get(), nullptr);

    EXPECT_NE(hipdnn_backend::platform_utilities::getSymbol(library.get(), "hipdnnPluginGetName"),
              nullptr);
}

TEST(TestPlatformUtils, GetSymbolClearsStaleDlerrorBeforeLookup)
{
    const LibraryHandleGuard library(
        hipdnn_backend::platform_utilities::openLibrary(TEST_PLUGIN_PATH));
    ASSERT_NE(library.get(), nullptr);

    EXPECT_EQ(
        hipdnn_data_sdk::utilities::getSymbol(library.get(), "hipdnnMissingSymbolForDlerrorTest"),
        nullptr);

    EXPECT_NE(hipdnn_backend::platform_utilities::getSymbol(library.get(), "hipdnnPluginGetName"),
              nullptr);
}

TEST(TestPlatformUtils, OpenLibraryThrowsHipdnnExceptionForMissingLibrary)
{
    EXPECT_THROW(hipdnn_backend::platform_utilities::openLibrary(
                     hipdnn_data_sdk::utilities::getLibraryName("hipdnn_missing_test_library")),
                 hipdnn_backend::HipdnnException);
}
