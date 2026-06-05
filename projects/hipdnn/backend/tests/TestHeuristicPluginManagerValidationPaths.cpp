// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPluginManagerValidationPaths.cpp
 * @brief Tests for HeuristicPluginManager validation code paths
 *
 * These tests load actual test plugins to exercise validateBeforeAdding() and
 * actionAfterAdding() to improve coverage of HeuristicPluginManager.hpp
 */

#include "HipdnnException.hpp"
#include "PlatformUtils.hpp"
#include "TestPluginConstants.hpp"
#include "plugin/HeuristicPluginManager.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

// Test plugin name constants (defined here because CMake ordering prevents proper macro propagation)
namespace
{
constexpr const char* BAD_API_VERSION_PLUGIN = "test_bad_api_version_heuristic_plugin";
constexpr const char* EMPTY_NAME_PLUGIN = "test_empty_name_heuristic_plugin";
constexpr const char* DUPLICATE_POLICY_ID_A_PLUGIN = "test_duplicate_policy_id_a_plugin";
constexpr const char* DUPLICATE_POLICY_ID_B_PLUGIN = "test_duplicate_policy_id_b_plugin";
} // namespace

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;
using namespace hipdnn_backend::plugin_constants;

class TestHeuristicPluginManagerValidationPaths : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        // Check once if test plugins are available
        const auto pluginPath = getHeuristicPluginPath("").parent_path();
        if(!std::filesystem::exists(pluginPath))
        {
            GTEST_SKIP() << "Test plugins not found at: " << pluginPath
                         << "\nMake sure test_plugins are built before running tests";
        }
    }

    void SetUp() override
    {
        // Test plugins are in lib/test_plugins/custom relative to backend library location
        _testPluginPath = getHeuristicPluginPath("").parent_path();

        // Create manager for each test
        _manager = std::make_unique<HeuristicPluginManager>();
    }

    std::unique_ptr<HeuristicPluginManager> _manager;
    std::filesystem::path _testPluginPath;
};

// ========== validateBeforeAdding() Tests - API Version Check ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, GoodPluginPassesApiVersionValidation)
{
    // Load good test plugin - should pass API version validation
    EXPECT_NO_THROW(_manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE));

    const auto& plugins = _manager->getPlugins();

    // Should have loaded at least the good plugin
    EXPECT_GT(plugins.size(), 0);

    // All loaded plugins should have correct API major version
    for(const auto& plugin : plugins)
    {
        const auto version = hipdnn_data_sdk::utilities::Version{plugin->apiVersion()};
        EXPECT_EQ(version.major, HIPDNN_HEURISTIC_API_VERSION_MAJOR)
            << "validateBeforeAdding should have checked API version for plugin: "
            << plugin->name();
    }
}

// ========== validateBeforeAdding() Tests - Policy ID Uniqueness ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, ActionAfterAddingStoresPolicyIds)
{
    // Load plugins - actionAfterAdding should store policy IDs
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const auto& plugins = _manager->getPlugins();

    // Collect all policy IDs to verify actionAfterAdding was called
    std::set<int64_t> policyIds;
    size_t totalPolicyCount = 0;
    for(const auto& plugin : plugins)
    {
        for(const int64_t id : plugin->getAllPolicyIds())
        {
            ++totalPolicyCount;

            // Each policy ID should be unique (actionAfterAdding should have tracked this)
            EXPECT_EQ(policyIds.count(id), 0) << "Policy ID " << id
                                              << " appears multiple times (actionAfterAdding "
                                                 "tracking failed)";

            policyIds.insert(id);
        }
    }

    // Should have loaded plugins with unique policy IDs
    EXPECT_GT(policyIds.size(), 0) << "Should have loaded at least one plugin policy";
    EXPECT_EQ(policyIds.size(), totalPolicyCount) << "All policy IDs should be unique";
}

TEST_F(TestHeuristicPluginManagerValidationPaths, PolicyIdTrackingAcrossMultiplePlugins)
{
    // Load all available plugins
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();

    // Collect all policy IDs from every loaded plugin/policy
    std::set<int64_t> policyIds;
    size_t totalPolicyCount = 0;
    for(const auto& plugin : plugins)
    {
        for(const int64_t id : plugin->getAllPolicyIds())
        {
            ++totalPolicyCount;

            // Each policy ID should be unique (actionAfterAdding tracks this)
            EXPECT_EQ(policyIds.count(id), 0)
                << "Policy ID " << id << " appears multiple times (validateBeforeAdding failed)";

            policyIds.insert(id);
        }
    }

    // Should have as many unique IDs as policies across all plugins
    EXPECT_EQ(policyIds.size(), totalPolicyCount);
}

// ========== validateBeforeAdding() Tests - Policy Name Check ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, AllLoadedPluginsHaveNonEmptyNames)
{
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();

    // validateBeforeAdding should have rejected any plugins with empty names
    for(const auto& plugin : plugins)
    {
        EXPECT_FALSE(plugin->name().empty())
            << "validateBeforeAdding should reject plugins with empty policy names";
    }
}

TEST_F(TestHeuristicPluginManagerValidationPaths, PolicyNameIsProvided)
{
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();
    EXPECT_GT(plugins.size(), 0);

    for(const auto& plugin : plugins)
    {
        // Plugin name must be non-empty (validated in HeuristicPluginManager)
        const std::string pluginName(plugin->name());
        EXPECT_FALSE(pluginName.empty()) << "Plugin has empty name";

        // Each policy must have a non-empty name (validated eagerly in HeuristicPlugin)
        for(const int64_t policyId : plugin->getAllPolicyIds())
        {
            const std::string policyName(plugin->getPolicyName(policyId));
            EXPECT_FALSE(policyName.empty()) << "Policy ID " << policyId << " has empty name";
        }
    }
}

// ========== Validation Success Path Tests ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, ValidPluginPassesAllValidation)
{
    // Load should succeed for valid plugins
    EXPECT_NO_THROW(_manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE));

    const auto& plugins = _manager->getPlugins();

    for(const auto& plugin : plugins)
    {
        // API version check
        const auto version = hipdnn_data_sdk::utilities::Version{plugin->apiVersion()};
        EXPECT_EQ(version.major, HIPDNN_HEURISTIC_API_VERSION_MAJOR);

        // Plugin name check
        EXPECT_FALSE(plugin->name().empty());

        // Each policy ID should be non-zero
        for(const int64_t policyId : plugin->getAllPolicyIds())
        {
            EXPECT_NE(policyId, 0);
        }
    }
}

// ========== actionAfterAdding() Coverage Tests ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, ActionAfterAddingExecutesForEachPlugin)
{
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();

    // For each policy across all plugins, actionAfterAdding should have inserted the
    // policy ID into _policyIds set. Verified indirectly by ensuring no duplicates exist.
    std::set<int64_t> observedIds;
    for(const auto& plugin : plugins)
    {
        for(const int64_t id : plugin->getAllPolicyIds())
        {
            EXPECT_EQ(observedIds.count(id), 0)
                << "Duplicate policy ID detected - actionAfterAdding may have failed";
            observedIds.insert(id);
        }
    }
}

// ========== Multiple Load Cycles with Validation ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, ValidationRunsOnEachLoadCycle)
{
    // Create new manager for each load to ensure fresh state
    HeuristicPluginManager manager1; // NOLINT(misc-const-correctness)
    manager1.loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const size_t count1 = manager1.getPlugins().size();

    // Second manager should also run validation and load same plugins
    HeuristicPluginManager manager2; // NOLINT(misc-const-correctness)
    manager2.loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const size_t count2 = manager2.getPlugins().size();

    EXPECT_EQ(count1, count2) << "Both managers should validate and load same plugins";
    EXPECT_GT(count1, 0) << "Should have loaded at least one plugin";
}

TEST_F(TestHeuristicPluginManagerValidationPaths, AbsoluteReloadResetsPolicyIdTracking)
{
    // Regression: ABSOLUTE-mode reload must clear the derived-class policy-id
    // index, not just _plugins. Otherwise reloading a plugin whose policy id
    // matches one from the previous load triggers a false "already exists"
    // failure in validateBeforeAdding.
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    const size_t firstCount = _manager->getPlugins().size();
    ASSERT_GT(firstCount, 0u) << "Test precondition: at least one plugin must load";

    EXPECT_NO_THROW(_manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE));
    EXPECT_EQ(_manager->getPlugins().size(), firstCount);
}

// ========== Constructor Path Coverage ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, ConstructorSetsUpValidationInfrastructure)
{
    // Constructor initializes with search paths and empty _policyIds set.
    // Assert against the freshly-constructed local manager (not the fixture's),
    // otherwise the test is just re-checking SetUp's invariant.
    const HeuristicPluginManager manager;

    // A freshly-constructed manager always contains the Config + StaticOrdering
    // built-ins (registered in HeuristicPluginManager's constructor); nothing else yet.
    EXPECT_EQ(manager.getPlugins().size(), 2u);
}

// ========== Destructor Path Coverage ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, DestructorUnloadsLoadedPluginLibraries)
{
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    ASSERT_FALSE(_manager->getPlugins().empty())
        << "Test precondition: at least one plugin must load to exercise the unload path";

    // Destruction triggers SharedLibrary teardown for each loaded plugin
    // (dlclose / FreeLibrary). ASAN catches any leak in plugin-side static teardown.
    EXPECT_NO_THROW(_manager.reset());
    EXPECT_EQ(_manager, nullptr);
}

// ========== Integration with PluginManagerBase ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, ValidationIntegratesWithBaseClass)
{
    // PluginManagerBase calls validateBeforeAdding before adding each plugin
    // and actionAfterAdding after successful add
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();

    // All plugins should have passed validation
    for(const auto& plugin : plugins)
    {
        // These checks verify that validateBeforeAdding was called and passed
        EXPECT_FALSE(plugin->name().empty());

        const auto version = hipdnn_data_sdk::utilities::Version{plugin->apiVersion()};
        EXPECT_EQ(version.major, HIPDNN_HEURISTIC_API_VERSION_MAJOR);

        for(const int64_t policyId : plugin->getAllPolicyIds())
        {
            EXPECT_NE(policyId, 0);
        }
    }
}

// ========== Specific Test Plugin Tests ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, NoOptionalHeuristicPluginPassesValidation)
{
    // test_no_optional_heuristic_plugin doesn't implement optional functions
    // but should still pass validation
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();
    ASSERT_FALSE(plugins.empty()) << "Expected at least one plugin to load from "
                                  << _testPluginPath;

    bool foundNoOptional = false;
    for(const auto& plugin : plugins)
    {
        const std::string name(plugin->name());
        if(name.find("NoOptional") != std::string::npos)
        {
            foundNoOptional = true;
            // Should pass all validation checks
            EXPECT_FALSE(name.empty());
            for(const int64_t policyId : plugin->getAllPolicyIds())
            {
                EXPECT_NE(policyId, 0);
            }
        }
    }
    EXPECT_TRUE(foundNoOptional) << "test_no_optional_heuristic_plugin should be loaded";
}

TEST_F(TestHeuristicPluginManagerValidationPaths, GoodHeuristicPluginPassesValidation)
{
    _manager->loadPlugins({_testPluginPath}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = _manager->getPlugins();
    ASSERT_FALSE(plugins.empty()) << "Expected at least one plugin to load from "
                                  << _testPluginPath;

    // Match the exact good-plugin name. A loose substring match would pass even if
    // the good plugin failed to load.
    constexpr const char* K_GOOD_PLUGIN_NAME = "TestGoodHeuristicPlugin";

    bool foundGood = false;
    for(const auto& plugin : plugins)
    {
        const std::string name(plugin->name());
        if(name == K_GOOD_PLUGIN_NAME)
        {
            foundGood = true;
            // Verify it passed all validation:
            // 1. API version
            const auto version = hipdnn_data_sdk::utilities::Version{plugin->apiVersion()};
            EXPECT_EQ(version.major, HIPDNN_HEURISTIC_API_VERSION_MAJOR);

            // 2. Plugin name is non-empty
            EXPECT_FALSE(name.empty());

            // 3. All policy IDs are unique and non-zero
            for(const int64_t policyId : plugin->getAllPolicyIds())
            {
                EXPECT_NE(policyId, 0);
            }
        }
    }
    EXPECT_TRUE(foundGood) << "test_good_heuristic_plugin should be loaded";
}

// ========== Validation Failure Tests ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, BadApiVersionPluginRejected)
{
    // ABSOLUTE mode accepts a single plugin file path, so we can load just the bad
    // plugin directly from the build tree instead of copying it to a temp dir.
    const auto badPlugin
        = _testPluginPath / hipdnn_data_sdk::utilities::getLibraryName(BAD_API_VERSION_PLUGIN);

    // Without this precondition, loadPlugins silently no-ops on a missing file
    // and the empty-plugins assertion below would pass vacuously.
    ASSERT_TRUE(std::filesystem::exists(badPlugin))
        << "Test precondition: bad-API-version plugin missing at " << badPlugin;

    _manager->loadPlugins({badPlugin}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    // Built-in Config + StaticOrdering are always present; no external plugin should have loaded.
    EXPECT_EQ(_manager->getPlugins().size(), 2u) << "Bad API version plugin should be rejected";
}

TEST_F(TestHeuristicPluginManagerValidationPaths, EmptyNamePluginRejected)
{
    const auto emptyNamePlugin
        = _testPluginPath / hipdnn_data_sdk::utilities::getLibraryName(EMPTY_NAME_PLUGIN);

    ASSERT_TRUE(std::filesystem::exists(emptyNamePlugin))
        << "Test precondition: empty-name plugin missing at " << emptyNamePlugin;

    _manager->loadPlugins({emptyNamePlugin}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    // Built-in Config + StaticOrdering are always present; no external plugin should have loaded.
    EXPECT_EQ(_manager->getPlugins().size(), 2u) << "Empty policy name plugin should be rejected";
}

TEST_F(TestHeuristicPluginManagerValidationPaths, DuplicatePolicyIdPluginsRejected)
{
    const auto pluginA = _testPluginPath
                         / hipdnn_data_sdk::utilities::getLibraryName(DUPLICATE_POLICY_ID_A_PLUGIN);
    const auto pluginB = _testPluginPath
                         / hipdnn_data_sdk::utilities::getLibraryName(DUPLICATE_POLICY_ID_B_PLUGIN);

    if(!std::filesystem::exists(pluginA) || !std::filesystem::exists(pluginB))
    {
        GTEST_SKIP() << "test_duplicate_policy_id plugins not found";
    }

    _manager->loadPlugins({pluginA, pluginB}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    // Built-in Config + StaticOrdering are always present, plus the first of the
    // duplicate pair (pluginA). The second (pluginB) is rejected for duplicate policy ID.
    const auto& plugins = _manager->getPlugins();
    ASSERT_EQ(plugins.size(), 3u) << "Built-ins + first duplicate plugin should be present";

    // The survivor must be pluginA (first offered). Probe pluginA on its own to
    // capture its policy IDs, then verify those IDs appear in the loaded set —
    // without this, the size check would still pass if pluginA was rejected for
    // an unrelated reason and pluginB loaded.
    HeuristicPluginManager probeA;
    probeA.loadPlugins({pluginA}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    ASSERT_EQ(probeA.getPlugins().size(), 3u)
        << "pluginA should load successfully alongside the built-ins to be a valid baseline";

    // Find the non-built-in plugin in the probe to get pluginA's policy IDs.
    std::vector<int64_t> pluginAPolicyIds;
    for(const auto& plugin : probeA.getPlugins())
    {
        const std::string name(plugin->name());
        if(name != "BuiltInStaticOrderingHeuristic" && name != "BuiltInConfigHeuristic")
        {
            pluginAPolicyIds = plugin->getAllPolicyIds();
            break;
        }
    }
    ASSERT_FALSE(pluginAPolicyIds.empty()) << "Probe failed to identify pluginA";

    bool foundPluginA = false;
    for(const auto& plugin : plugins)
    {
        if(plugin->getAllPolicyIds() == pluginAPolicyIds)
        {
            foundPluginA = true;
            break;
        }
    }
    EXPECT_TRUE(foundPluginA) << "Survivor should be pluginA (the first offered), not pluginB";
}

// ========== loadPluginFromFile Return-Value Regression ==========

// Expose loadPluginFromFile() so we can directly observe its bool return.
// The bug it guards against: success was set to true at the top of the
// tryCatch lambda, so a throwing validateBeforeAdding() (e.g. bad API
// version) left success == true and the caller's failedCount silently
// stayed at zero.
class LoadPluginFromFileProbe : public HeuristicPluginManager
{
public:
    using HeuristicPluginManager::loadPluginFromFile;
};

TEST_F(TestHeuristicPluginManagerValidationPaths, LoadPluginFromFileReturnsFalseOnValidationFailure)
{
    const auto badPlugin
        = _testPluginPath / hipdnn_data_sdk::utilities::getLibraryName(BAD_API_VERSION_PLUGIN);
    ASSERT_TRUE(std::filesystem::exists(badPlugin))
        << "Test precondition: bad-API-version plugin missing at " << badPlugin;

    LoadPluginFromFileProbe probe;
    const size_t pluginCountBefore = probe.getPlugins().size();

    EXPECT_FALSE(probe.loadPluginFromFile(badPlugin))
        << "loadPluginFromFile must report failure when validateBeforeAdding throws";
    EXPECT_EQ(probe.getPlugins().size(), pluginCountBefore)
        << "Rejected plugin must not be appended to _plugins";
}

TEST_F(TestHeuristicPluginManagerValidationPaths, LoadPluginFromFileReturnsTrueOnSuccess)
{
    const auto goodPlugin = getHeuristicPluginPath("test_good_heuristic_plugin");
    ASSERT_TRUE(std::filesystem::exists(goodPlugin))
        << "Test precondition: good plugin missing at " << goodPlugin;

    LoadPluginFromFileProbe probe;
    const size_t pluginCountBefore = probe.getPlugins().size();

    EXPECT_TRUE(probe.loadPluginFromFile(goodPlugin));
    EXPECT_EQ(probe.getPlugins().size(), pluginCountBefore + 1u);

    // A second load of the same file is an idempotent no-op (already in
    // _loadedPluginFiles); it must also return true so failedCount is
    // not inflated by retries.
    EXPECT_TRUE(probe.loadPluginFromFile(goodPlugin))
        << "Idempotent reload of an already-loaded plugin must not count as failure";
    EXPECT_EQ(probe.getPlugins().size(), pluginCountBefore + 1u);
}

// ========== Edge Case: Empty Plugin Directory ==========

TEST_F(TestHeuristicPluginManagerValidationPaths, EmptyDirectorySkipsValidation)
{
    // ScopedDirectory creates the dir and remove_all's it on destruction, so
    // the temp dir is cleaned up even if an assertion below aborts.
    const auto uniqueName = std::string("hipdnn_empty_heur_test_")
                            + std::to_string(::testing::UnitTest::GetInstance()->random_seed());
    const hipdnn_test_sdk::utilities::ScopedDirectory emptyDir(
        std::filesystem::temp_directory_path() / uniqueName);

    // Load from empty directory - no plugins to validate
    _manager->loadPlugins({emptyDir.path()}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    // Built-in Config + StaticOrdering are always present; the empty directory contributed nothing.
    EXPECT_EQ(_manager->getPlugins().size(), 2u);
}
