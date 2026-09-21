// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <test_plugins/TestPluginConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_tests::plugin_constants;
namespace fs = std::filesystem;

// Check if any loaded path has a filename matching the expected path's filename.
// Sufficient for these tests since all test plugins have unique filenames.
static bool containsPluginByFilename(const std::vector<fs::path>& loadedPaths,
                                     const std::string& expectedPath)
{
    fs::path expectedFilename = fs::path(expectedPath).filename();
    return std::any_of(
        loadedPaths.begin(), loadedPaths.end(), [&expectedFilename](const fs::path& loaded) {
            return loaded.filename() == expectedFilename;
        });
}

// Check if any loaded path resides in the same folder as expectedPath.
static bool containsPluginByFolder(const std::vector<fs::path>& loadedPaths,
                                   const std::string& expectedPath)
{
    fs::path expectedFolder = fs::path(expectedPath).parent_path();
    return std::any_of(
        loadedPaths.begin(), loadedPaths.end(), [&expectedFolder](const fs::path& loaded) {
            return loaded.parent_path() == expectedFolder;
        });
}

TEST(IntegrationFrontendSetPluginPathsExt, EmptyPathsAdditive)
{
    // Reset plugin paths from any prior test to ensure clean state
    const std::vector<fs::path> emptyPaths = {};
    setEnginePluginPaths(emptyPaths, PluginLoadingMode::MODE_ABSOLUTE);

    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_PLUGIN_DIR", getTestPluginDefaultDir());

    auto error = setEnginePluginPaths(emptyPaths, PluginLoadingMode::MODE_ADDITIVE);
    ASSERT_TRUE(error.is_good());

    auto [handle, err] = createHipdnnHandle();
    ASSERT_TRUE(err.is_good());

    std::vector<fs::path> loadedPaths;
    error = getLoadedEnginePluginPaths(*handle, loadedPaths);
    ASSERT_TRUE(error.is_good());

    const std::string expectedPluginPath = testDefaultGoodPluginPath();

    EXPECT_EQ(loadedPaths.size(), 1);
    EXPECT_TRUE(containsPluginByFilename(loadedPaths, expectedPluginPath));
}

TEST(IntegrationFrontendSetPluginPathsExt, AbsoluteLoadsOnlyCustom)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_PLUGIN_DIR", getTestPluginDefaultDir());

    const auto& pluginFilePath = testGoodPluginPath();
    const std::array<const char*, 1> paths = {pluginFilePath.c_str()};

    auto error = setEnginePluginPaths(paths, PluginLoadingMode::MODE_ABSOLUTE);
    ASSERT_TRUE(error.is_good());

    auto [handle, err] = createHipdnnHandle();
    ASSERT_TRUE(err.is_good());

    std::vector<fs::path> loadedPaths;
    error = getLoadedEnginePluginPaths(*handle, loadedPaths);
    ASSERT_TRUE(error.is_good());

    EXPECT_EQ(loadedPaths.size(), 1);

    auto defaultPluginPath
        = (fs::path("hipdnn_plugins/engines")
           / hipdnn_data_sdk::utilities::getLibraryName("test_good_default_plugin"))
              .string();

    EXPECT_FALSE(containsPluginByFolder(loadedPaths, defaultPluginPath));
    EXPECT_FALSE(containsPluginByFolder(loadedPaths, testDefaultGoodPluginPath()));
    EXPECT_TRUE(containsPluginByFilename(loadedPaths, testGoodPluginPath()));
}

TEST(IntegrationFrontendSetPluginPathsExt, AdditiveLoadsBothDefaultAndCustom)
{
    // Reset plugin paths from any prior test to ensure clean state
    setEnginePluginPaths(std::vector<fs::path>{}, PluginLoadingMode::MODE_ABSOLUTE);

    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_PLUGIN_DIR", getTestPluginDefaultDir());

    const std::array<const char*, 1> paths = {getTestPluginCustomDir().c_str()};
    auto error = setEnginePluginPaths(paths, PluginLoadingMode::MODE_ADDITIVE);
    ASSERT_TRUE(error.is_good());

    auto [handle, err] = createHipdnnHandle();
    ASSERT_TRUE(err.is_good());

    std::vector<fs::path> loadedPaths;
    error = getLoadedEnginePluginPaths(*handle, loadedPaths);
    ASSERT_TRUE(error.is_good());

    EXPECT_GE(loadedPaths.size(), 2);

    auto defaultPluginPath = testDefaultGoodPluginPath();
    const auto& testPluginPath = testGoodPluginPath();

    EXPECT_TRUE(containsPluginByFilename(loadedPaths, defaultPluginPath));
    EXPECT_TRUE(containsPluginByFilename(loadedPaths, testPluginPath));
}
