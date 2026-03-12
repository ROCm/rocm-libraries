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

TEST(IntegrationFrontendSetPluginPathsExt, EmptyPathsAdditive)
{
    // Reset plugin paths from any prior test to ensure clean state
    std::vector<fs::path> emptyPaths = {};
    setEnginePluginPaths(emptyPaths, PluginLoadingMode::ABSOLUTE);

    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_PLUGIN_DIR", getTestPluginDefaultDir());

    auto error = setEnginePluginPaths(emptyPaths, PluginLoadingMode::ADDITIVE);
    ASSERT_TRUE(error.is_good());

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    std::vector<fs::path> loadedPaths;
    error = getLoadedEnginePluginPaths(handle, loadedPaths);
    ASSERT_TRUE(error.is_good());

    std::string expectedPluginPath = getDefaultPluginPath();

    EXPECT_EQ(loadedPaths.size(), 1);
    EXPECT_TRUE(containsPluginByFilename(loadedPaths, expectedPluginPath));
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationFrontendSetPluginPathsExt, AbsoluteLoadsOnlyCustom)
{
    const auto& pluginFilePath = testGoodPluginPath();
    std::array<const char*, 1> paths = {pluginFilePath.c_str()};

    auto error = setEnginePluginPaths(paths, PluginLoadingMode::ABSOLUTE);
    ASSERT_TRUE(error.is_good());

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    std::vector<fs::path> loadedPaths;
    error = getLoadedEnginePluginPaths(handle, loadedPaths);
    ASSERT_TRUE(error.is_good());

    EXPECT_EQ(loadedPaths.size(), 1);

    auto defaultPluginPath
        = fs::path("hipdnn_plugins/engines")
          / hipdnn_data_sdk::utilities::getLibraryName("test_good_default_plugin");
    const auto& testPluginPath = testGoodPluginPath();

    EXPECT_FALSE(containsPluginByFilename(loadedPaths, defaultPluginPath.string()));
    EXPECT_TRUE(containsPluginByFilename(loadedPaths, testPluginPath));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationFrontendSetPluginPathsExt, AdditiveLoadsBothDefaultAndCustom)
{
    // Reset plugin paths from any prior test to ensure clean state
    setEnginePluginPaths(std::vector<fs::path>{}, PluginLoadingMode::ABSOLUTE);

    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_PLUGIN_DIR", getTestPluginDefaultDir());

    const std::array<const char*, 1> paths = {getTestPluginCustomDir().c_str()};
    auto error = setEnginePluginPaths(paths, PluginLoadingMode::ADDITIVE);
    ASSERT_TRUE(error.is_good());

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    std::vector<fs::path> loadedPaths;
    error = getLoadedEnginePluginPaths(handle, loadedPaths);
    ASSERT_TRUE(error.is_good());

    EXPECT_GE(loadedPaths.size(), 2);

    auto defaultPluginPath = getDefaultPluginPath();
    const auto& testPluginPath = testGoodPluginPath();

    EXPECT_TRUE(containsPluginByFilename(loadedPaths, defaultPluginPath));
    EXPECT_TRUE(containsPluginByFilename(loadedPaths, testPluginPath));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationFrontendGetPluginPathsExt, GetLoadedPluginPathsAfterAbsolute)
{
    const auto& pluginFilePath = testGoodPluginPath();
    std::array<const char*, 1> setPaths = {pluginFilePath.c_str()};

    auto error = setEnginePluginPaths(setPaths, PluginLoadingMode::ABSOLUTE);
    ASSERT_TRUE(error.is_good());

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    std::vector<fs::path> loadedPaths;
    error = getLoadedEnginePluginPaths(handle, loadedPaths);
    ASSERT_TRUE(error.is_good());

    EXPECT_EQ(loadedPaths.size(), 1);
    EXPECT_TRUE(containsPluginByFilename(loadedPaths, pluginFilePath));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationFrontendGetPluginPathsExt, GetLoadedPluginPathsAfterAdditive)
{
    // Reset plugin paths from any prior test to ensure clean state
    setEnginePluginPaths(std::vector<fs::path>{}, PluginLoadingMode::ABSOLUTE);

    hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_PLUGIN_DIR", getTestPluginDefaultDir());

    const std::array<const char*, 1> paths = {getTestPluginCustomDir().c_str()};
    auto error = setEnginePluginPaths(paths, PluginLoadingMode::ADDITIVE);
    ASSERT_TRUE(error.is_good());

    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    std::vector<fs::path> loadedPaths;
    error = getLoadedEnginePluginPaths(handle, loadedPaths);
    ASSERT_TRUE(error.is_good());

    EXPECT_GE(loadedPaths.size(), 2);

    auto defaultPluginPath = getDefaultPluginPath();
    const auto& testPluginPath = testGoodPluginPath();

    EXPECT_TRUE(containsPluginByFilename(loadedPaths, defaultPluginPath));
    EXPECT_TRUE(containsPluginByFilename(loadedPaths, testPluginPath));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
