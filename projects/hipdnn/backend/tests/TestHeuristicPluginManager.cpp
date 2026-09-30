// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPluginManager.cpp
 * @brief Unit tests for HeuristicPluginManager validation logic
 *
 * These tests verify the plugin discovery and validation layer including:
 * - API version compatibility validation
 * - Policy ID uniqueness validation
 * - Policy ID ↔ policy name consistency validation
 */

#include "HipdnnException.hpp"
#include "plugin/HeuristicPlugin.hpp"
#include "plugin/HeuristicPluginManager.hpp"

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include <filesystem>
#include <gtest/gtest.h>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;

class TestHeuristicPluginManager : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Most tests use validation logic, not actual plugin loading
    }

    void TearDown() override {}

    // Helper to create a valid policy name/ID pair
    static std::pair<std::string, int64_t> makeValidPolicyPair(const std::string& baseName)
    {
        const int64_t policyId = hipdnn_data_sdk::utilities::policyNameToId(baseName);
        return {baseName, policyId};
    }
};

// ========== Construction Tests ==========

TEST_F(TestHeuristicPluginManager, ConstructorSucceeds)
{
    EXPECT_NO_THROW(const HeuristicPluginManager manager);
}

// ========== Plugin Loading Tests ==========

TEST_F(TestHeuristicPluginManager, LoadPluginsFromEmptyDirectorySucceeds)
{
    HeuristicPluginManager manager;

    const auto uniqueName = std::string("hipdnn_test_empty_")
                            + std::to_string(::testing::UnitTest::GetInstance()->random_seed());
    const hipdnn_test_sdk::utilities::ScopedDirectory emptyDir(
        std::filesystem::temp_directory_path() / uniqueName);

    EXPECT_NO_THROW(manager.loadPlugins({emptyDir.path()}, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
}

TEST_F(TestHeuristicPluginManager, LoadPluginsFromNonexistentDirectorySucceeds)
{
    HeuristicPluginManager manager;

    const std::filesystem::path nonexistentDir
        = std::filesystem::temp_directory_path() / "nonexistent_path_to_plugins";

    // Should not throw - just logs warning and continues
    EXPECT_NO_THROW(manager.loadPlugins({nonexistentDir}, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
}

TEST_F(TestHeuristicPluginManager, LoadPluginsWithMultiplePathsSucceeds)
{
    HeuristicPluginManager manager;

    const std::set<std::filesystem::path> paths
        = {std::filesystem::temp_directory_path() / "path1",
           std::filesystem::temp_directory_path() / "path2",
           std::filesystem::temp_directory_path() / "path3"};

    // Should not throw
    EXPECT_NO_THROW(manager.loadPlugins(paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
}

// Note: Actual validation happens inside validateBeforeAdding() which is protected.
// It is exercised end-to-end in TestHeuristicPluginManagerValidationPaths.cpp using
// real test plugins.

// ========== Multiple Load Cycles Tests ==========

TEST_F(TestHeuristicPluginManager, MultipleLoadCyclesSucceed)
{
    HeuristicPluginManager manager;

    const std::set<std::filesystem::path> paths
        = {std::filesystem::temp_directory_path() / "test_plugins"};

    // Load multiple times
    for(int i = 0; i < 5; ++i)
    {
        EXPECT_NO_THROW(manager.loadPlugins(paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
    }
}

TEST_F(TestHeuristicPluginManager, LoadingDifferentPathsSucceeds)
{
    HeuristicPluginManager manager;

    // Load from path 1
    EXPECT_NO_THROW(manager.loadPlugins({std::filesystem::temp_directory_path() / "path1"},
                                        HIPDNN_PLUGIN_LOADING_ABSOLUTE));

    // Load from path 2
    EXPECT_NO_THROW(manager.loadPlugins({std::filesystem::temp_directory_path() / "path2"},
                                        HIPDNN_PLUGIN_LOADING_ABSOLUTE));
}

// ========== Thread Safety Tests ==========

TEST_F(TestHeuristicPluginManager, MultipleInstancesAreIndependent)
{
    HeuristicPluginManager manager1;
    HeuristicPluginManager manager2;

    // Load different paths
    manager1.loadPlugins({std::filesystem::temp_directory_path() / "path1"},
                         HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    manager2.loadPlugins({std::filesystem::temp_directory_path() / "path2"},
                         HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    // Only the always-registered Config + StaticOrdering built-ins remain; no
    // external plugin loaded from a non-existent path.
    EXPECT_EQ(manager1.getPlugins().size(), 2u);
    EXPECT_EQ(manager2.getPlugins().size(), 2u);
}

// ========== Edge Cases Tests ==========

TEST_F(TestHeuristicPluginManager, LoadPluginsWithEmptyPathListSucceeds)
{
    HeuristicPluginManager manager;

    const std::set<std::filesystem::path> emptyPaths;

    // Should not throw
    EXPECT_NO_THROW(manager.loadPlugins(emptyPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
}

TEST_F(TestHeuristicPluginManager, LoadPluginsWithSamePathTwiceSucceeds)
{
    HeuristicPluginManager manager;

    const std::set<std::filesystem::path> paths = {std::filesystem::temp_directory_path() / "test"};

    // Set automatically handles duplicates, but test that loading twice works
    manager.loadPlugins(paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    EXPECT_NO_THROW(manager.loadPlugins(paths, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
}

// ========== Plugin Loading Mode Tests ==========

TEST_F(TestHeuristicPluginManager, LoadPluginsWithAbsoluteModeResets)
{
    HeuristicPluginManager manager;

    const std::set<std::filesystem::path> paths1
        = {std::filesystem::temp_directory_path() / "path1"};
    const std::set<std::filesystem::path> paths2
        = {std::filesystem::temp_directory_path() / "path2"};

    manager.loadPlugins(paths1, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const size_t count1 = manager.getPlugins().size();

    manager.loadPlugins(paths2, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const size_t count2 = manager.getPlugins().size();

    // Both should be 0 since paths don't exist, but operation should succeed
    EXPECT_EQ(count1, count2);
}

TEST_F(TestHeuristicPluginManager, LoadPluginsWithAdditiveModeAdds)
{
    HeuristicPluginManager manager;

    const std::set<std::filesystem::path> paths1
        = {std::filesystem::temp_directory_path() / "path1"};
    const std::set<std::filesystem::path> paths2
        = {std::filesystem::temp_directory_path() / "path2"};

    // Exercise the additive code path. With non-existent paths the plugin count
    // stays at 0, but the additive branch in PluginManagerBase::loadPlugins is
    // executed and any leak/crash inside it would be caught under ASAN.
    EXPECT_NO_THROW(manager.loadPlugins(paths1, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
    EXPECT_NO_THROW(manager.loadPlugins(paths2, HIPDNN_PLUGIN_LOADING_ADDITIVE));
}

// Destructor coverage with real loaded plugins lives in
// TestHeuristicPluginManagerValidationPaths.cpp::DestructorUnloadsLoadedPluginLibraries,
// where _testPluginPath provides actual shared libraries to exercise the unload path.

// ========== Policy ID/Name Validation Tests ==========

TEST_F(TestHeuristicPluginManager, PolicyNameToIdIsConsistent)
{
    const std::string name1 = "Vendor::PolicyA";
    const int64_t id1a = hipdnn_data_sdk::utilities::policyNameToId(name1);
    const int64_t id1b = hipdnn_data_sdk::utilities::policyNameToId(name1);

    EXPECT_EQ(id1a, id1b);
}

TEST_F(TestHeuristicPluginManager, DifferentNamesProduceDifferentIds)
{
    const std::string name1 = "Vendor::PolicyA";
    const std::string name2 = "SelectionHeuristic::StaticOrdering";

    const int64_t id1 = hipdnn_data_sdk::utilities::policyNameToId(name1);
    const int64_t id2 = hipdnn_data_sdk::utilities::policyNameToId(name2);

    EXPECT_NE(id1, id2);
}

TEST_F(TestHeuristicPluginManager, PolicyIdIsNonZero)
{
    const std::string name = "Vendor::PolicyA";
    const int64_t id = hipdnn_data_sdk::utilities::policyNameToId(name);

    EXPECT_NE(id, 0);
}

TEST_F(TestHeuristicPluginManager, EmptyPolicyNameProducesZeroId)
{
    const std::string emptyName;
    const int64_t id = hipdnn_data_sdk::utilities::policyNameToId(emptyName);

    // Empty string should produce a specific ID (FNV-1a hash of empty string)
    EXPECT_EQ(id, 0);
}

// ========== State Verification Tests ==========

TEST_F(TestHeuristicPluginManager, GetPluginsAfterEmptyLoadReturnsEmpty)
{
    HeuristicPluginManager manager;

    const auto uniqueName = std::string("hipdnn_empty_load_test_")
                            + std::to_string(::testing::UnitTest::GetInstance()->random_seed());
    const hipdnn_test_sdk::utilities::ScopedDirectory emptyDir(
        std::filesystem::temp_directory_path() / uniqueName);

    manager.loadPlugins({emptyDir.path()}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    // Only the always-registered Config + StaticOrdering built-ins remain; the
    // empty dir contributed nothing.
    EXPECT_EQ(manager.getPlugins().size(), 2u);
}
