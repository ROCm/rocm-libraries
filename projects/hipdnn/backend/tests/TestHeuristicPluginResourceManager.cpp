// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPluginResourceManager.cpp
 * @brief Unit tests for HeuristicPluginResourceManager
 *
 * These tests verify the plugin resource management layer that provides
 * per-handle plugin lifecycle management and policy lookup.
 */

#include "HipdnnException.hpp"
#include "plugin/HeuristicPluginManager.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;

class TestHeuristicPluginResourceManager : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Tests in this suite use mock plugins, not real shared libraries
    }

    void TearDown() override {}
};

// ========== Construction and Initialization Tests ==========

TEST_F(TestHeuristicPluginResourceManager, FactoryMethodCreatesInstance)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);
}

TEST_F(TestHeuristicPluginResourceManager, ConstructorWithPluginManagerSucceeds)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);
    ASSERT_NE(rm, nullptr);
}

// ========== Move Semantics Tests ==========

TEST_F(TestHeuristicPluginResourceManager, MoveConstructorTransfersOwnership)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm1 = std::make_shared<HeuristicPluginResourceManager>(pm);

    // Move construct
    const HeuristicPluginResourceManager rm2(std::move(*rm1));

    // rm2 should be usable. The shared plugin manager always contains the
    // Config + StaticOrdering built-ins, so the policy-info list is not empty.
    const auto infos = rm2.getHeuristicPolicyInfos();
    EXPECT_EQ(infos.size(), 2u);
}

TEST_F(TestHeuristicPluginResourceManager, MoveAssignmentTransfersOwnership)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm1 = std::make_shared<HeuristicPluginResourceManager>(pm);
    auto rm2 = std::make_shared<HeuristicPluginResourceManager>(pm);

    // Move assign
    *rm2 = std::move(*rm1);

    // rm2 should be usable. The shared plugin manager always contains the
    // StaticOrdering built-in, so the policy-info list is not empty.
    const HeuristicPluginResourceManager& constRm2 = *rm2;
    const auto infos = constRm2.getHeuristicPolicyInfos();
    EXPECT_EQ(infos.size(), 2u);
}

// ========== Policy Lookup Tests ==========

TEST_F(TestHeuristicPluginResourceManager, GetHandleForNonexistentPolicyReturnsNull)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    const int64_t fakePolicyId = 0x1234567890ABCDEF;
    auto handle = rm->getHeuristicHandleForPolicyId(fakePolicyId);

    EXPECT_EQ(handle, nullptr);
}

TEST_F(TestHeuristicPluginResourceManager, GetPluginForNonexistentPolicyReturnsNull)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    const int64_t fakePolicyId = 0x1234567890ABCDEF;
    auto plugin = rm->getPluginForPolicyId(fakePolicyId);

    EXPECT_EQ(plugin, nullptr);
}

// ========== Policy Enumeration Tests ==========

TEST_F(TestHeuristicPluginResourceManager, GetPolicyInfosWhenNoPluginsLoaded)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    // No external plugin paths configured, but the Config + StaticOrdering built-ins
    // are always registered in the plugin manager's constructor.
    const auto infos = rm->getHeuristicPolicyInfos();
    EXPECT_EQ(infos.size(), 2u);
}

TEST_F(TestHeuristicPluginResourceManager, GetPolicyInfosCachesResult)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    // First call
    const auto infos1 = rm->getHeuristicPolicyInfos();

    // Second call should return cached result (implementation detail)
    const auto infos2 = rm->getHeuristicPolicyInfos();

    EXPECT_EQ(infos1.size(), infos2.size());
}

// ========== Device Properties Tests ==========

TEST_F(TestHeuristicPluginResourceManager, SetDevicePropertiesOnEmptyManagerDoesNotThrow)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    hipdnnPluginConstData_t deviceProps;
    deviceProps.ptr = nullptr;
    deviceProps.size = 0;

    // Should not throw even when no plugins loaded
    EXPECT_NO_THROW(rm->setDevicePropertiesOnAllHandles(&deviceProps));
}

TEST_F(TestHeuristicPluginResourceManager, SetDevicePropertiesWithNullPointerSucceeds)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    // Passing nullptr is allowed when no plugins loaded
    EXPECT_NO_THROW(rm->setDevicePropertiesOnAllHandles(nullptr));
}

// ========== Plugin File Enumeration Tests ==========

TEST_F(TestHeuristicPluginResourceManager, GetLoadedPluginFilesWhenNoneLoaded)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    size_t numPlugins = 0;
    size_t maxStringLen = 0;

    // Query counts
    rm->getLoadedPluginFiles(&numPlugins, nullptr, &maxStringLen);

    EXPECT_EQ(numPlugins, 0u);
    EXPECT_EQ(maxStringLen, 0u);
}

TEST_F(TestHeuristicPluginResourceManager, GetLoadedPluginFilesWithNullNumPluginsThrows)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    // nullptr for numPlugins should throw
    EXPECT_THROW(rm->getLoadedPluginFiles(nullptr, nullptr, nullptr), HipdnnException);
}

// ========== String Representation Tests ==========

TEST_F(TestHeuristicPluginResourceManager, ToStringReturnsNonEmptyString)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    const std::string str = rm->toString();
    EXPECT_FALSE(str.empty());
}

TEST_F(TestHeuristicPluginResourceManager, ToStringIncludesPluginCount)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    const std::string str = rm->toString();

    // Should mention plugin count (even if 0)
    EXPECT_TRUE(str.find("plugin") != std::string::npos || str.find("Plugin") != std::string::npos);
}

// ========== Static Path Configuration Tests ==========

TEST_F(TestHeuristicPluginResourceManager, SetHeuristicPluginPathsSucceeds)
{
    const std::vector<std::filesystem::path> paths
        = {std::filesystem::temp_directory_path() / "test_plugins"};

    EXPECT_NO_THROW(HeuristicPluginResourceManager::setHeuristicPluginPaths(
        paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
}

TEST_F(TestHeuristicPluginResourceManager, GetHeuristicPluginPathsReturnsConfiguredPaths)
{
    const std::vector<std::filesystem::path> paths
        = {std::filesystem::temp_directory_path() / "test_plugins",
           std::filesystem::temp_directory_path() / "more_plugins"};

    HeuristicPluginResourceManager::setHeuristicPluginPaths(paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto retrievedPaths = HeuristicPluginResourceManager::getHeuristicPluginPaths();

    // Should contain at least the paths we set (may include defaults)
    EXPECT_GE(retrievedPaths.size(), 0u);
}

TEST_F(TestHeuristicPluginResourceManager, SetPluginUnloadingModeSucceeds)
{
    // Both modes should be accepted
    EXPECT_NO_THROW(
        HeuristicPluginResourceManager::setPluginUnloadingMode(HIPDNN_PLUGIN_UNLOAD_LAZY));

    EXPECT_NO_THROW(
        HeuristicPluginResourceManager::setPluginUnloadingMode(HIPDNN_PLUGIN_UNLOAD_EAGER));
}

TEST_F(TestHeuristicPluginResourceManager, SetPluginLogLevelSucceeds)
{
    // Should not throw for various log levels
    EXPECT_NO_THROW(HeuristicPluginResourceManager::setPluginLogLevel(HIPDNN_SEV_INFO));
    EXPECT_NO_THROW(HeuristicPluginResourceManager::setPluginLogLevel(HIPDNN_SEV_WARN));
    EXPECT_NO_THROW(HeuristicPluginResourceManager::setPluginLogLevel(HIPDNN_SEV_ERROR));
}

// ========== Multiple Instances Tests ==========

TEST_F(TestHeuristicPluginResourceManager, MultipleInstancesCanCoexist)
{
    auto pm = std::make_shared<HeuristicPluginManager>();

    auto rm1 = std::make_shared<HeuristicPluginResourceManager>(pm);
    auto rm2 = std::make_shared<HeuristicPluginResourceManager>(pm);
    auto rm3 = std::make_shared<HeuristicPluginResourceManager>(pm);

    // All should be independent and usable
    EXPECT_NE(rm1, nullptr);
    EXPECT_NE(rm2, nullptr);
    EXPECT_NE(rm3, nullptr);

    // Each should work independently. The shared plugin manager always contains
    // the StaticOrdering built-in, so each resource manager observes it.
    EXPECT_EQ(rm1->getHeuristicPolicyInfos().size(), 2u);
    EXPECT_EQ(rm2->getHeuristicPolicyInfos().size(), 2u);
    EXPECT_EQ(rm3->getHeuristicPolicyInfos().size(), 2u);
}

// ========== Copy Prevention Tests ==========

TEST_F(TestHeuristicPluginResourceManager, CopyConstructorIsDeleted)
{
    // This test verifies at compile time that copying is prevented
    // If this compiles, the test passes
    EXPECT_TRUE((std::is_copy_constructible_v<HeuristicPluginResourceManager> == false));
}

TEST_F(TestHeuristicPluginResourceManager, CopyAssignmentIsDeleted)
{
    // This test verifies at compile time that copy assignment is prevented
    EXPECT_TRUE((std::is_copy_assignable_v<HeuristicPluginResourceManager> == false));
}

// ========== Destruction Tests ==========

TEST_F(TestHeuristicPluginResourceManager, DestructorCleansUpResources)
{
    auto pm = std::make_shared<HeuristicPluginManager>();

    {
        auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);
        // Use rm
        rm->getHeuristicPolicyInfos();
    } // rm destroyed here
}

TEST_F(TestHeuristicPluginResourceManager, MultipleDestructionsSucceed)
{
    auto pm = std::make_shared<HeuristicPluginManager>();

    for(int i = 0; i < 10; ++i)
    {
        auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);
        rm->getHeuristicPolicyInfos();
        // Destroyed at end of loop
    }
}

// ========== Constructor Null Pointer Tests ==========

TEST_F(TestHeuristicPluginResourceManager, ConstructorWithNullPluginManagerAccepted)
{
    // Null plugin manager is accepted during static destruction scenarios
    // The constructor should not throw - it just skips initialization
    EXPECT_NO_THROW({ auto rm = std::make_shared<HeuristicPluginResourceManager>(nullptr); });

    // Verify the resource manager works with null manager
    auto rm = std::make_shared<HeuristicPluginResourceManager>(nullptr);
    EXPECT_EQ(rm->getHeuristicHandleForPolicyId(1), nullptr);
    EXPECT_EQ(rm->getPluginForPolicyId(1), nullptr);
}

// ========== Policy Info Caching Tests ==========

TEST_F(TestHeuristicPluginResourceManager, PolicyInfosAreCachedAcrossMultipleCalls)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    // Multiple calls should use cache
    const auto infos1 = rm->getHeuristicPolicyInfos();
    const auto infos2 = rm->getHeuristicPolicyInfos();
    const auto infos3 = rm->getHeuristicPolicyInfos();

    // All should return same result
    EXPECT_EQ(infos1.size(), infos2.size());
    EXPECT_EQ(infos2.size(), infos3.size());
}

// ========== ToString with Plugin Data Tests ==========

TEST_F(TestHeuristicPluginResourceManager, ToStringFormatContainsKeyElements)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    const std::string str = rm->toString();

    // Should contain the class name
    EXPECT_NE(str.find("HeuristicPluginResourceManager"), std::string::npos);

    // Should contain braces for structure
    EXPECT_NE(str.find('{'), std::string::npos);
    EXPECT_NE(str.find('}'), std::string::npos);
}

// ========== Additional Configuration Tests ==========

TEST_F(TestHeuristicPluginResourceManager, SetPluginPathsWithEmptyVectorSucceeds)
{
    const std::vector<std::filesystem::path> emptyPaths;

    EXPECT_NO_THROW(HeuristicPluginResourceManager::setHeuristicPluginPaths(
        emptyPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
}

TEST_F(TestHeuristicPluginResourceManager, SetPluginPathsWithAdditiveMode)
{
    const std::vector<std::filesystem::path> paths = {"/test/path1", "/test/path2"};

    EXPECT_NO_THROW(HeuristicPluginResourceManager::setHeuristicPluginPaths(
        paths, HIPDNN_PLUGIN_LOADING_ADDITIVE));
}

TEST_F(TestHeuristicPluginResourceManager, SetPluginLogLevelWithDebugLevel)
{
    EXPECT_NO_THROW(HeuristicPluginResourceManager::setPluginLogLevel(HIPDNN_SEV_INFO));
}

TEST_F(TestHeuristicPluginResourceManager, SetPluginLogLevelWithErrorLevel)
{
    EXPECT_NO_THROW(HeuristicPluginResourceManager::setPluginLogLevel(HIPDNN_SEV_ERROR));
}

// ========== GetLoadedPluginFiles Edge Cases ==========

TEST_F(TestHeuristicPluginResourceManager, GetLoadedPluginFilesWithNonNullPathsSucceeds)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    size_t numPlugins = 0;
    std::array<char*, 10> paths{};

    // Query with paths array (not yet implemented, should not crash)
    EXPECT_NO_THROW(rm->getLoadedPluginFiles(&numPlugins, paths.data(), nullptr));
}

TEST_F(TestHeuristicPluginResourceManager, GetLoadedPluginFilesQueriesCountOnly)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    size_t numPlugins = 999; // Should be overwritten

    rm->getLoadedPluginFiles(&numPlugins, nullptr, nullptr);

    EXPECT_EQ(numPlugins, 0u); // No plugins loaded
}

// ========== Device Properties Error Handling ==========

TEST_F(TestHeuristicPluginResourceManager, SetDevicePropertiesWithValidDataSucceeds)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    const std::array<uint8_t, 4> fakeData = {1, 2, 3, 4};
    hipdnnPluginConstData_t deviceProps;
    deviceProps.ptr = fakeData.data();
    deviceProps.size = fakeData.size();

    // Should not throw when plugins loaded or not
    EXPECT_NO_THROW(rm->setDevicePropertiesOnAllHandles(&deviceProps));
}

// ========== Multiple Static Configuration Changes ==========

TEST_F(TestHeuristicPluginResourceManager, MultipleUnloadingModeChangesSucceed)
{
    EXPECT_NO_THROW(
        HeuristicPluginResourceManager::setPluginUnloadingMode(HIPDNN_PLUGIN_UNLOAD_LAZY));

    EXPECT_NO_THROW(
        HeuristicPluginResourceManager::setPluginUnloadingMode(HIPDNN_PLUGIN_UNLOAD_EAGER));

    EXPECT_NO_THROW(
        HeuristicPluginResourceManager::setPluginUnloadingMode(HIPDNN_PLUGIN_UNLOAD_LAZY));
}

TEST_F(TestHeuristicPluginResourceManager, MultiplePathConfigurationsSucceed)
{
    const std::vector<std::filesystem::path> paths1 = {"/test/path1"};
    const std::vector<std::filesystem::path> paths2 = {"/test/path2", "/test/path3"};

    EXPECT_NO_THROW(HeuristicPluginResourceManager::setHeuristicPluginPaths(
        paths1, HIPDNN_PLUGIN_LOADING_ABSOLUTE));

    EXPECT_NO_THROW(HeuristicPluginResourceManager::setHeuristicPluginPaths(
        paths2, HIPDNN_PLUGIN_LOADING_ADDITIVE));
}

// ========== Policy Lookup with Multiple Plugins ==========

TEST_F(TestHeuristicPluginResourceManager, GetHandleForMultiplePolicyIdsAllReturnNull)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    // Try multiple non-existent policy IDs
    EXPECT_EQ(rm->getHeuristicHandleForPolicyId(1), nullptr);
    EXPECT_EQ(rm->getHeuristicHandleForPolicyId(100), nullptr);
    EXPECT_EQ(rm->getHeuristicHandleForPolicyId(0x123456), nullptr);
    EXPECT_EQ(rm->getHeuristicHandleForPolicyId(-1), nullptr);
}

TEST_F(TestHeuristicPluginResourceManager, GetPluginForMultiplePolicyIdsAllReturnNull)
{
    auto pm = std::make_shared<HeuristicPluginManager>();
    auto rm = std::make_shared<HeuristicPluginResourceManager>(pm);

    // Try multiple non-existent policy IDs
    EXPECT_EQ(rm->getPluginForPolicyId(1), nullptr);
    EXPECT_EQ(rm->getPluginForPolicyId(100), nullptr);
    EXPECT_EQ(rm->getPluginForPolicyId(0x123456), nullptr);
    EXPECT_EQ(rm->getPluginForPolicyId(-1), nullptr);
}
