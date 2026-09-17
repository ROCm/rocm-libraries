// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <unordered_set>

#include "HipdnnException.hpp"
#include "PlatformUtils.hpp"
#include "TestPluginConstants.hpp"
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>

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

TEST(TestPlatformUtils, GenerateUuidV4ProducesValidVersion4Uuid)
{
    const auto uuid = hipdnn_backend::platform_utilities::generateUuidV4();
    EXPECT_TRUE(hipdnn_flatbuffers_sdk::utilities::isUuidV4(uuid));
}

TEST(TestPlatformUtils, GenerateUuidV4ProducesDistinctValues)
{
    const auto first = hipdnn_backend::platform_utilities::generateUuidV4();
    const auto second = hipdnn_backend::platform_utilities::generateUuidV4();
    EXPECT_NE(first, second);
}

TEST(TestPlatformUtils, GenerateUuidV4ProducesManyDistinctValues)
{
    // A broken RNG (fixed seed, zeroed buffer, stale entropy pool) would collapse
    // this set well below 1000; a healthy one should never collide at this scale.
    constexpr int UUID_COUNT = 1000;
    std::unordered_set<std::string> seen;
    for(int i = 0; i < UUID_COUNT; ++i)
    {
        const auto uuid = hipdnn_backend::platform_utilities::generateUuidV4();
        EXPECT_TRUE(hipdnn_flatbuffers_sdk::utilities::isUuidV4(uuid));
        seen.insert(hipdnn_flatbuffers_sdk::utilities::formatUuid(uuid));
    }
    EXPECT_EQ(seen.size(), static_cast<size_t>(UUID_COUNT));
}
