// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "../test_plugins/TestPluginConstants.hpp"
#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/test_utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <vector>

using namespace hipdnn_sdk::utilities;
using namespace hipdnn_tests::plugin_constants;
namespace fs = std::filesystem;

TEST(IntegrationSetPluginPathsExt, ValidInputs)
{
    std::array<const char*, 3> paths = {getTestPluginCustomDir().c_str(), "./", "../directory/"};

    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationSetPluginPathsExt, InvalidAndValidNullptrCorrectness)
{
    hipdnnStatus_t status
        = hipdnnSetEnginePluginPaths_ext(1, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    status = hipdnnSetEnginePluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto loadedPlugins = test_util::getLoadedPlugins(handle);
    EXPECT_EQ(loadedPlugins.size(), 0);

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationSetPluginPathsExt, NullStringInList)
{
    std::array<const char*, 2> paths = {"./valid/path.so", nullptr};

    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(IntegrationSetPluginPathsExt, IneligibleHandle)
{
    hipdnnHandle_t handle = nullptr;
    hipdnnStatus_t status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    const std::array<const char*, 1> paths = {"./fake/path"};
    status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    EXPECT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);

    size_t numPlugins = 0;
    size_t maxPathLength = 0;
    status = hipdnnGetLoadedEnginePluginPaths_ext(nullptr, &numPlugins, nullptr, &maxPathLength);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(IntegrationSetPluginPathsExt, GetLoadedPluginPathsLoadsDefault)
{
    hipdnn_sdk::test_utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_PLUGIN_DIR", getTestPluginDefaultDir());

    hipdnnStatus_t status
        = hipdnnSetEnginePluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto loadedPlugins = test_util::getLoadedPlugins(handle);

    std::string expectedPluginPath = getDefaultPluginPath();

    EXPECT_EQ(loadedPlugins.size(), 1);
    EXPECT_TRUE(test_util::isPluginLoadedByRelativePath(loadedPlugins, expectedPluginPath));
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationSetPluginPathsExt, GetLoadedPluginPathsAdditiveLoadsBothDefaultAndCustom)
{
    hipdnn_sdk::test_utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_PLUGIN_DIR", getTestPluginDefaultDir());

    const std::array<const char*, 1> paths = {getTestPluginCustomDir().c_str()};
    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ADDITIVE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto loadedPlugins = test_util::getLoadedPlugins(handle);
    EXPECT_GE(loadedPlugins.size(), 2);

    auto defaultPluginPath = getDefaultPluginPath();
    const auto& testPluginPath = testGoodPluginPath();

    EXPECT_TRUE(test_util::isPluginLoadedByRelativePath(loadedPlugins, defaultPluginPath));
    EXPECT_TRUE(test_util::isPluginLoadedByRelativePath(loadedPlugins, testPluginPath));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationSetPluginPathsExt, GetLoadedPluginPathsAbsoluteLoadsOnlyCustom)
{
    const auto& pluginFilePath = testGoodPluginPath();
    const std::array<const char*, 1> paths = {pluginFilePath.c_str()};
    hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto loadedPlugins = test_util::getLoadedPlugins(handle);
    EXPECT_EQ(loadedPlugins.size(), 1);

    auto defaultPluginPath
        = fs::path("hipdnn_plugins/engines") / getLibraryName("test_good_default_plugin");
    const auto& testPluginPath = testGoodPluginPath();

    EXPECT_FALSE(test_util::isPluginLoadedByRelativePath(loadedPlugins, defaultPluginPath));
    EXPECT_TRUE(test_util::isPluginLoadedByRelativePath(loadedPlugins, testPluginPath));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
