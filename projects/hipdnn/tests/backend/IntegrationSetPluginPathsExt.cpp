// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "../test_plugins/TestPluginConstants.hpp"
#include "TestUtil.hpp"
#include "hipdnn_backend.h"
#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_tests::plugin_constants;
namespace fs = std::filesystem;

namespace
{
struct HeuristicPolicyInfoForTest
{
    std::string policyName;
    std::string pluginName;
};

bool getHeuristicPolicyInfos(hipdnnHandle_t handle,
                             std::vector<HeuristicPolicyInfoForTest>& policyInfos)
{
    size_t numPolicies = 0;
    auto status = hipdnnGetHeuristicPolicyCount_ext(handle, &numPolicies);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        ADD_FAILURE() << "Failed to query heuristic policy count: " << status;
        return false;
    }

    policyInfos.clear();
    policyInfos.reserve(numPolicies);

    for(size_t i = 0; i < numPolicies; ++i)
    {
        size_t policyNameLen = 0;
        size_t pluginNameLen = 0;
        size_t pluginVersionLen = 0;
        size_t apiVersionLen = 0;
        status = hipdnnGetHeuristicPolicyInfo_ext(handle,
                                                  i,
                                                  nullptr,
                                                  nullptr,
                                                  &policyNameLen,
                                                  nullptr,
                                                  &pluginNameLen,
                                                  nullptr,
                                                  &pluginVersionLen,
                                                  nullptr,
                                                  &apiVersionLen);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            ADD_FAILURE() << "Failed to query heuristic policy info sizes for index " << i << ": "
                          << status;
            return false;
        }

        if(policyNameLen == 0 || pluginNameLen == 0 || pluginVersionLen == 0 || apiVersionLen == 0)
        {
            ADD_FAILURE() << "Heuristic policy info for index " << i
                          << " reported an empty metadata field";
            return false;
        }

        std::vector<char> policyName(policyNameLen);
        std::vector<char> pluginName(pluginNameLen);
        std::vector<char> pluginVersion(pluginVersionLen);
        std::vector<char> apiVersion(apiVersionLen);
        status = hipdnnGetHeuristicPolicyInfo_ext(handle,
                                                  i,
                                                  nullptr,
                                                  policyName.data(),
                                                  &policyNameLen,
                                                  pluginName.data(),
                                                  &pluginNameLen,
                                                  pluginVersion.data(),
                                                  &pluginVersionLen,
                                                  apiVersion.data(),
                                                  &apiVersionLen);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            ADD_FAILURE() << "Failed to retrieve heuristic policy info for index " << i << ": "
                          << status;
            return false;
        }

        policyInfos.push_back({policyName.data(), pluginName.data()});
    }

    return true;
}

bool heuristicPolicyIsLoaded(hipdnnHandle_t handle, const char* expectedPolicyName)
{
    std::vector<HeuristicPolicyInfoForTest> policyInfos;
    if(!getHeuristicPolicyInfos(handle, policyInfos))
    {
        return false;
    }

    for(const auto& policyInfo : policyInfos)
    {
        if(policyInfo.policyName == expectedPolicyName)
        {
            return true;
        }
    }

    return false;
}

bool onlyBuiltInHeuristicPoliciesAreLoaded(hipdnnHandle_t handle)
{
    std::vector<HeuristicPolicyInfoForTest> policyInfos;
    if(!getHeuristicPolicyInfos(handle, policyInfos))
    {
        return false;
    }

    for(const auto& policyInfo : policyInfos)
    {
        if(policyInfo.pluginName.rfind("BuiltIn", 0) != 0)
        {
            ADD_FAILURE() << "Loaded path-discovered heuristic policy '" << policyInfo.policyName
                          << "' from plugin '" << policyInfo.pluginName << "'";
            return false;
        }
    }

    return true;
}
} // namespace

TEST(IntegrationSetPluginPathsExt, ValidInputs)
{
    std::array<const char*, 3> paths = {getTestPluginCustomDir().c_str(), "./", "../directory/"};

    const hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
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

TEST(IntegrationSetPluginPathsExt, HeuristicInvalidAndValidNullptrCorrectness)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_HEURISTIC_PLUGIN_DIR",
        fs::path(testGoodHeuristicPluginPath()).parent_path().string());

    hipdnnStatus_t status
        = hipdnnSetHeuristicPluginPaths_ext(1, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);

    status = hipdnnSetHeuristicPluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    // Built-in heuristic policies are always registered. Empty ABSOLUTE paths should suppress
    // only path-discovered external heuristic plugins, including the env-sourced test plugin.
    EXPECT_TRUE(onlyBuiltInHeuristicPoliciesAreLoaded(handle));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnSetHeuristicPluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE),
              HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationSetPluginPathsExt, HeuristicNullStringInList)
{
    std::array<const char*, 2> paths = {testGoodHeuristicPluginPath().c_str(), nullptr};

    const hipdnnStatus_t status = hipdnnSetHeuristicPluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    EXPECT_EQ(status, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(IntegrationSetPluginPathsExt, HeuristicEmptyAdditiveLoadsDefault)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_HEURISTIC_PLUGIN_DIR",
        fs::path(testGoodHeuristicPluginPath()).parent_path().string());

    hipdnnStatus_t status
        = hipdnnSetHeuristicPluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    status = hipdnnSetHeuristicPluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    EXPECT_TRUE(heuristicPolicyIsLoaded(handle, testGoodHeuristicPolicyName()));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnSetHeuristicPluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ABSOLUTE),
              HIPDNN_STATUS_SUCCESS);
    EXPECT_EQ(hipdnnSetHeuristicPluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE),
              HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationSetPluginPathsExt, NullStringInList)
{
    std::array<const char*, 2> paths = {"./valid/path.so", nullptr};

    const hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
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

TEST(IntegrationSetPluginPathsExt, HeuristicIneligibleHandle)
{
    hipdnnHandle_t handle = nullptr;
    hipdnnStatus_t status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    const std::array<const char*, 1> paths = {testGoodHeuristicPluginPath().c_str()};
    status = hipdnnSetHeuristicPluginPaths_ext(
        paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    EXPECT_EQ(status, HIPDNN_STATUS_NOT_SUPPORTED);

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationSetPluginPathsExt, GetLoadedPluginPathsLoadsDefault)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envSetter(
        "HIPDNN_PLUGIN_DIR", getTestPluginDefaultDir());

    hipdnnStatus_t status
        = hipdnnSetEnginePluginPaths_ext(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);

    hipdnnHandle_t handle = nullptr;
    status = hipdnnCreate(&handle);
    ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
    ASSERT_NE(handle, nullptr);

    auto loadedPlugins = test_util::getLoadedPlugins(handle);

    const std::string expectedPluginPath = testDefaultGoodPluginPath();

    EXPECT_EQ(loadedPlugins.size(), 1);
    EXPECT_TRUE(test_util::isPluginLoadedByRelativePath(loadedPlugins, expectedPluginPath));
    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

TEST(IntegrationSetPluginPathsExt, GetLoadedPluginPathsAdditiveLoadsBothDefaultAndCustom)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter envSetter(
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

    auto defaultPluginPath = testDefaultGoodPluginPath();
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

    EXPECT_FALSE(
        test_util::isPluginLoadedByRelativePath(loadedPlugins, defaultPluginPath.string()));
    EXPECT_TRUE(test_util::isPluginLoadedByRelativePath(loadedPlugins, testPluginPath));

    EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
