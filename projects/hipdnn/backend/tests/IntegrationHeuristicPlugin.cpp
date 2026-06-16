// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file IntegrationHeuristicPlugin.cpp
 * @brief Integration tests for HeuristicPlugin workflow coverage
 *
 * These tests exercise full workflows with real plugins. The file holds four
 * sibling fixtures, each focused on a distinct concept:
 *  - IntegrationHeuristicPlugin: resource-manager-mediated paths against a
 *    directory of test plugins.
 *  - IntegrationHeuristicPluginLoadedGood: direct construction of the good
 *    test plugin to exercise the HeuristicPlugin object surface.
 *  - IntegrationHeuristicPluginLoadedNoOptional: direct construction of a
 *    plugin missing optional symbols, verifying graceful degradation.
 *  - IntegrationHeuristicPluginIncomplete: a plugin missing required symbols
 *    must be rejected at construction time.
 */

#include "HipdnnException.hpp"
#include "PlatformUtils.hpp"
#include "TestPluginConstants.hpp"
#include "heuristics/DeviceProperties.hpp"
#include "plugin/HeuristicPlugin.hpp"
#include "plugin/HeuristicPluginManager.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"
#include "plugin/SharedLibrary.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/device_properties_generated.h>

#include <gtest/gtest.h>
#include <string_view>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;
using namespace hipdnn_backend::plugin_constants;

namespace
{
// Note: TEST_GOOD_HEURISTIC_PLUGIN_NAME, TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME,
// and TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME are defined as macros in CMakeLists.txt

using hipdnn_backend::heuristics::serializeDeviceProperties;
using hipdnn_backend::heuristics::wrapSerializedDeviceProperties;

// Wrapper class to access protected constructor
class TestableHeuristicPlugin : public HeuristicPlugin
{
public:
    explicit TestableHeuristicPlugin(SharedLibrary&& lib)
        : HeuristicPlugin(std::move(lib))
    {
    }
};

// RAII guards for plugin opaque handles. The handle and policy descriptor are
// allocated by the plugin's C ABI (`createHandle` / `createPolicyDescriptor`)
// and must be released via the matching destroy call. Without a guard, any
// ASSERT_* abort or thrown exception between create and destroy leaks under
// ASAN. Built on top of the shared ScopedResource<T> utility.
inline auto makeScopedHandle(const HeuristicPlugin& plugin, hipdnnHeuristicHandle_t handle)
{
    return hipdnn_data_sdk::utilities::ScopedResource<hipdnnHeuristicHandle_t>(
        handle, [p = &plugin](hipdnnHeuristicHandle_t h) {
            if(h != nullptr)
            {
                p->destroyHandle(h);
            }
        });
}

inline auto makeScopedPolicyDescriptor(const HeuristicPlugin& plugin,
                                       hipdnnHeuristicPolicyDescriptor_t desc)
{
    return hipdnn_data_sdk::utilities::ScopedResource<hipdnnHeuristicPolicyDescriptor_t>(
        desc, [p = &plugin](hipdnnHeuristicPolicyDescriptor_t d) {
            if(d != nullptr)
            {
                p->destroyPolicyDescriptor(d);
            }
        });
}

} // namespace

// ====================================================================================
// IntegrationHeuristicPlugin: resource-manager-mediated workflows
// ====================================================================================

class IntegrationHeuristicPlugin : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Set plugin path to test plugins directory
        const auto testPluginDir = getHeuristicPluginPath("").parent_path();
        HeuristicPluginResourceManager::setHeuristicPluginPaths({testPluginDir},
                                                                HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    }

    void TearDown() override
    {
        // Reset to default empty paths
        HeuristicPluginResourceManager::setHeuristicPluginPaths({}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
    }
};

// ========== Complete Workflow Tests ==========

TEST_F(IntegrationHeuristicPlugin, CompleteHandleLifecycleWithGoodPlugin)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    // Should have loaded test plugins
    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    // Look up the good test plugin by its known policy id rather than scanning
    // — getPluginForPolicyId is non-null for any id sourced from
    // getHeuristicPolicyInfos(), so a "find the first non-null" loop is dead.
    const auto goodPolicyId = hipdnn_data_sdk::utilities::policyNameToId("TestGoodHeuristicPolicy");
    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(goodPolicyId);
    hipdnnHeuristicHandle_t handle = rm->getHeuristicHandleForPolicyId(goodPolicyId);

    ASSERT_NE(plugin, nullptr);
    ASSERT_NE(handle, nullptr);

    // Verify plugin metadata is available
    EXPECT_FALSE(plugin->version().empty());
}

// ========== Basic Operation Tests ==========
// Note: Basic individual operations (createHandle, createPolicyDescriptor, setEngineIds,
// setSerializedGraph, finalize, getSortedEngineIds) are tested in IntegrationHeuristicPluginLoadedGood
// fixture with focused assertions. Tests here focus on resource manager integration.

TEST_F(IntegrationHeuristicPlugin, SetDevicePropertiesOnHandle)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    // policyInfos ordering comes from an unordered_map iteration; target the
    // test plugin by known policy ID so behavior is stable across platforms.
    const auto goodPolicyId = hipdnn_data_sdk::utilities::policyNameToId("TestGoodHeuristicPolicy");
    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(goodPolicyId);
    ASSERT_NE(plugin, nullptr);

    hipdnnHeuristicHandle_t handle = rm->getHeuristicHandleForPolicyId(goodPolicyId);
    ASSERT_NE(handle, nullptr);

    // Create device properties
    hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT props;
    props.device_id = 0;
    props.multi_processor_count = 120;
    props.total_global_mem = 16ULL * 1024 * 1024 * 1024; // 16 GB
    props.architecture_name = "gfx90a";

    // Serialize
    const auto serialized = serializeDeviceProperties(props);
    const hipdnnPluginConstData_t devicePropsData = wrapSerializedDeviceProperties(serialized);

    // Set on handle (should not throw)
    EXPECT_NO_THROW(plugin->setDeviceProperties(handle, &devicePropsData));
}

TEST_F(IntegrationHeuristicPlugin, SetDevicePropertiesOnAllHandles)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    // Create device properties
    hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT props;
    props.device_id = 0;
    props.multi_processor_count = 120;
    props.total_global_mem = 16ULL * 1024 * 1024 * 1024;
    props.architecture_name = "gfx90a";

    const auto serialized = serializeDeviceProperties(props);
    const hipdnnPluginConstData_t devicePropsData = wrapSerializedDeviceProperties(serialized);

    // Set on all handles via resource manager
    EXPECT_NO_THROW(rm->setDevicePropertiesOnAllHandles(&devicePropsData));
}

TEST_F(IntegrationHeuristicPlugin, CompleteWorkflowWithDevicePropertiesAndFinalize)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    // Target the test plugin by known policy ID; built-ins reject the
    // synthetic graph payload below.
    const auto goodPolicyId = hipdnn_data_sdk::utilities::policyNameToId("TestGoodHeuristicPolicy");
    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(goodPolicyId);
    ASSERT_NE(plugin, nullptr);

    hipdnnHeuristicHandle_t handle = rm->getHeuristicHandleForPolicyId(goodPolicyId);
    ASSERT_NE(handle, nullptr);

    // Set device properties on handle
    hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT props;
    props.device_id = 0;
    props.multi_processor_count = 120;
    props.total_global_mem = 16ULL * 1024 * 1024 * 1024;
    props.architecture_name = "gfx90a";

    const auto serialized = serializeDeviceProperties(props);
    const hipdnnPluginConstData_t devicePropsData = wrapSerializedDeviceProperties(serialized);
    plugin->setDeviceProperties(handle, &devicePropsData);

    // Create policy descriptor (RAII so destroy runs even on ASSERT_* abort)
    const auto descGuard
        = makeScopedPolicyDescriptor(*plugin, plugin->createPolicyDescriptor(handle, goodPolicyId));
    ASSERT_NE(descGuard.get(), nullptr);

    // Set engine IDs
    const std::vector<int64_t> engineIds = {1000, 2000, 3000};
    plugin->setEngineIds(descGuard.get(), engineIds.data(), engineIds.size());

    // Set serialized graph
    const std::vector<uint8_t> graphBytes = {10, 20, 30};
    hipdnnPluginConstData_t serializedGraph;
    serializedGraph.ptr = graphBytes.data();
    serializedGraph.size = graphBytes.size();
    plugin->setSerializedGraph(descGuard.get(), &serializedGraph);

    // Finalize
    plugin->finalize(descGuard.get());

    // Get results
    const auto sortedIds = plugin->getSortedEngineIds(descGuard.get());
}

// ========== Plugin Metadata Coverage ==========
// Note: Plugin metadata queries (name, version, API version, policy ID) are tested
// in IntegrationHeuristicPluginLoadedGood fixture with more specific assertions

TEST_F(IntegrationHeuristicPlugin, GetPluginTypeFromLoadedPlugin)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    // policyInfos ordering comes from an unordered_map iteration; target the
    // test plugin by known policy ID so behavior is stable across platforms.
    const auto goodPolicyId = hipdnn_data_sdk::utilities::policyNameToId("TestGoodHeuristicPolicy");
    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(goodPolicyId);
    ASSERT_NE(plugin, nullptr);

    // Heuristic plugins report HEURISTIC type
    const auto pluginType = plugin->type();
    EXPECT_EQ(pluginType, HIPDNN_PLUGIN_TYPE_HEURISTIC);
}

// ========== Resource Manager Enumeration Coverage ==========

TEST_F(IntegrationHeuristicPlugin, GetLoadedPluginFilesReturnsCorrectCount)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    size_t numPlugins = 0;
    size_t maxStringLen = 0;

    rm->getLoadedPluginFiles(&numPlugins, nullptr, &maxStringLen);

    // Should have at least the test plugins
    EXPECT_GT(numPlugins, 0u);
}

TEST_F(IntegrationHeuristicPlugin, ToStringContainsPluginInformation)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto str = rm->toString();

    EXPECT_NE(str.find("HeuristicPluginResourceManager"), std::string::npos);
    EXPECT_NE(str.find("Loaded plugins:"), std::string::npos);
}

// ========== Multiple Descriptors Per Handle ==========

TEST_F(IntegrationHeuristicPlugin, MultipleDescriptorsFromSameHandle)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    // policyInfos ordering comes from an unordered_map iteration; target the
    // test plugin by known policy ID so behavior is stable across platforms.
    const auto policyId = hipdnn_data_sdk::utilities::policyNameToId("TestGoodHeuristicPolicy");
    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(policyId);
    ASSERT_NE(plugin, nullptr);

    hipdnnHeuristicHandle_t handle = rm->getHeuristicHandleForPolicyId(policyId);
    ASSERT_NE(handle, nullptr);

    // Create multiple descriptors from the same handle (RAII-wrapped so any
    // assertion abort below still releases them).
    const auto desc1
        = makeScopedPolicyDescriptor(*plugin, plugin->createPolicyDescriptor(handle, policyId));
    const auto desc2
        = makeScopedPolicyDescriptor(*plugin, plugin->createPolicyDescriptor(handle, policyId));
    const auto desc3
        = makeScopedPolicyDescriptor(*plugin, plugin->createPolicyDescriptor(handle, policyId));

    EXPECT_NE(desc1.get(), nullptr);
    EXPECT_NE(desc2.get(), nullptr);
    EXPECT_NE(desc3.get(), nullptr);

    // Note: Test plugins may return the same hardcoded pointer for simplicity,
    // but real plugins should return distinct descriptors. We just verify they're created.
}

// ========== Error Path Tests ==========
// These tests exercise error handling and edge cases

// ========== Error Path: Device Properties Exceptions ==========

TEST_F(IntegrationHeuristicPlugin, SetDevicePropertiesHandlesPluginFailures)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    // Create invalid device properties (empty buffer)
    hipdnnPluginConstData_t invalidProps;
    invalidProps.ptr = nullptr;
    invalidProps.size = 0;

    // Should not throw even if some plugins fail - logs warning and continues
    EXPECT_NO_THROW(rm->setDevicePropertiesOnAllHandles(&invalidProps));
}

// ========== Error Path: Missing Optional Functions ==========

TEST_F(IntegrationHeuristicPlugin, SetPluginLogLevelHandlesMissingOptionalFunction)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    // setPluginLogLevel should not throw even if optional function is missing
    EXPECT_NO_THROW(rm->setPluginLogLevel(HIPDNN_SEV_INFO));
}

// ========== Error Path: Empty Engine IDs ==========

TEST_F(IntegrationHeuristicPlugin, FinalizeWithEmptyEngineIdsSucceeds)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    // Target the test plugin by known policy ID; built-ins reject finalize()
    // without a real graph payload.
    const auto goodPolicyId = hipdnn_data_sdk::utilities::policyNameToId("TestGoodHeuristicPolicy");
    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(goodPolicyId);
    ASSERT_NE(plugin, nullptr);

    hipdnnHeuristicHandle_t handle = rm->getHeuristicHandleForPolicyId(goodPolicyId);
    ASSERT_NE(handle, nullptr);

    const auto descGuard
        = makeScopedPolicyDescriptor(*plugin, plugin->createPolicyDescriptor(handle, goodPolicyId));
    ASSERT_NE(descGuard.get(), nullptr);

    // Don't set any engine IDs - just finalize
    plugin->finalize(descGuard.get());

    // Get sorted IDs (should be empty)
    const auto sortedIds = plugin->getSortedEngineIds(descGuard.get());
    EXPECT_TRUE(sortedIds.empty());
}

// ========== Error Path: Multiple Policy Lookups (Same Handle/Plugin Reuse) ==========

TEST_F(IntegrationHeuristicPlugin, MultipleGetHandleCallsReturnSameHandle)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    // Pin to a known policy ID so the test does not depend on the
    // unordered_map iteration order behind getHeuristicPolicyInfos().
    const auto policyId = hipdnn_data_sdk::utilities::policyNameToId("TestGoodHeuristicPolicy");

    // Multiple calls should return the same handle (cached)
    auto handle1 = rm->getHeuristicHandleForPolicyId(policyId);
    auto handle2 = rm->getHeuristicHandleForPolicyId(policyId);

    EXPECT_EQ(handle1, handle2);
    EXPECT_NE(handle1, nullptr);
}

TEST_F(IntegrationHeuristicPlugin, MultipleGetPluginCallsReturnSamePlugin)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    // Pin to a known policy ID so the test does not depend on the
    // unordered_map iteration order behind getHeuristicPolicyInfos().
    const auto policyId = hipdnn_data_sdk::utilities::policyNameToId("TestGoodHeuristicPolicy");

    // Multiple calls should return the same plugin pointer
    const HeuristicPlugin* plugin1 = rm->getPluginForPolicyId(policyId);
    const HeuristicPlugin* plugin2 = rm->getPluginForPolicyId(policyId);

    EXPECT_EQ(plugin1, plugin2);
    EXPECT_NE(plugin1, nullptr);
}

// ========== Error Path: No plugins loaded scenario ==========

TEST_F(IntegrationHeuristicPlugin, SetDevicePropertiesWithNoPluginsLoaded)
{
    // Create RM with no plugins
    HeuristicPluginResourceManager::setHeuristicPluginPaths({}, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT props;
    props.device_id = 0;
    props.multi_processor_count = 120;
    props.total_global_mem = 16ULL * 1024 * 1024 * 1024;
    props.architecture_name = "gfx90a";

    const auto serialized = serializeDeviceProperties(props);
    const hipdnnPluginConstData_t devicePropsData = wrapSerializedDeviceProperties(serialized);

    // Should not throw when no plugins loaded
    EXPECT_NO_THROW(rm->setDevicePropertiesOnAllHandles(&devicePropsData));
}

// ====================================================================================
// IntegrationHeuristicPluginLoadedGood: direct construction of the good test plugin
// ====================================================================================

class IntegrationHeuristicPluginLoadedGood : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto pluginPath = getHeuristicPluginPath(TEST_GOOD_HEURISTIC_PLUGIN_NAME);
        ASSERT_TRUE(std::filesystem::exists(pluginPath))
            << "Test plugin not found: " << pluginPath
            << "\nMake sure test_plugins are built before running tests";

        SharedLibrary lib(pluginPath);
        _pluginPtr = std::make_unique<TestableHeuristicPlugin>(std::move(lib));
    }

    void TearDown() override
    {
        _pluginPtr.reset();
    }

    TestableHeuristicPlugin& plugin()
    {
        return *_pluginPtr;
    }

private:
    std::unique_ptr<TestableHeuristicPlugin> _pluginPtr;
};

TEST_F(IntegrationHeuristicPluginLoadedGood, LoadedPluginCanQueryApiVersion)
{
    const auto version = plugin().apiVersion();
    EXPECT_FALSE(version.empty());
    EXPECT_EQ(version, HIPDNN_HEURISTIC_API_VERSION);
}

TEST_F(IntegrationHeuristicPluginLoadedGood, LoadedPluginCanQueryPolicyId)
{
    const auto policyIds = plugin().getAllPolicyIds();
    ASSERT_EQ(policyIds.size(), 1u);
    const auto expectedId = hipdnn_data_sdk::utilities::policyNameToId("TestGoodHeuristicPolicy");
    EXPECT_EQ(policyIds.front(), expectedId);
}

TEST_F(IntegrationHeuristicPluginLoadedGood, LoadedPluginCanQueryPluginName)
{
    const auto name = plugin().name();
    EXPECT_EQ(name, "TestGoodHeuristicPlugin");
}

TEST_F(IntegrationHeuristicPluginLoadedGood, LoadedPluginCanQueryPolicyName)
{
    const auto policyIds = plugin().getAllPolicyIds();
    ASSERT_EQ(policyIds.size(), 1u);
    const auto policyName = plugin().getPolicyName(policyIds.front());
    EXPECT_EQ(policyName, "TestGoodHeuristicPolicy");
}

TEST_F(IntegrationHeuristicPluginLoadedGood, LoadedPluginCanQueryPluginVersion)
{
    const auto version = plugin().version();
    EXPECT_EQ(version, "1.0.0");
}

TEST_F(IntegrationHeuristicPluginLoadedGood, LoadedPluginCanGetSortedEngineIds)
{
    const auto handleGuard = makeScopedHandle(plugin(), plugin().createHandle());
    const auto policyId = plugin().getAllPolicyIds().front();
    const auto descGuard = makeScopedPolicyDescriptor(
        plugin(), plugin().createPolicyDescriptor(handleGuard.get(), policyId));

    const std::vector<int64_t> inputIds = {1, 2, 3, 4, 5};
    plugin().setEngineIds(descGuard.get(), inputIds.data(), inputIds.size());
    plugin().finalize(descGuard.get());

    std::vector<int64_t> sortedIds;
    ASSERT_NO_THROW({ sortedIds = plugin().getSortedEngineIds(descGuard.get()); });

    // Good plugin reverses the order
    EXPECT_EQ(sortedIds.size(), inputIds.size());
    EXPECT_EQ(sortedIds, std::vector<int64_t>({5, 4, 3, 2, 1}));
}

TEST_F(IntegrationHeuristicPluginLoadedGood, RealPluginCachesPolicyIds)
{
    // First call - IDs are queried from the plugin
    const auto ids1 = plugin().getAllPolicyIds();
    ASSERT_EQ(ids1.size(), 1u);
    const auto expectedId = hipdnn_data_sdk::utilities::policyNameToId("TestGoodHeuristicPolicy");
    EXPECT_EQ(ids1.front(), expectedId);

    // Second call should return the cached vector
    const auto ids2 = plugin().getAllPolicyIds();
    EXPECT_EQ(ids2, ids1);
}

// ====================================================================================
// IntegrationHeuristicPluginLoadedNoOptional: plugin missing optional symbols
// ====================================================================================

class IntegrationHeuristicPluginLoadedNoOptional : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto pluginPath = getHeuristicPluginPath(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME);
        ASSERT_TRUE(std::filesystem::exists(pluginPath))
            << "Test plugin not found: " << pluginPath
            << "\nMake sure test_plugins are built before running tests";

        SharedLibrary lib(pluginPath);
        _pluginPtr = std::make_unique<TestableHeuristicPlugin>(std::move(lib));
    }

    void TearDown() override
    {
        _pluginPtr.reset();
    }

    TestableHeuristicPlugin& plugin()
    {
        return *_pluginPtr;
    }

private:
    std::unique_ptr<TestableHeuristicPlugin> _pluginPtr;
};

TEST_F(IntegrationHeuristicPluginLoadedNoOptional, PluginWithoutOptionalPolicyNameHasName)
{
    // hipdnnPluginGetName is required
    const auto name = plugin().name();
    EXPECT_FALSE(name.empty());
    EXPECT_EQ(name, "TestNoOptionalHeuristicPlugin");

    // Each policy has a non-empty name
    for(const int64_t policyId : plugin().getAllPolicyIds())
    {
        EXPECT_FALSE(plugin().getPolicyName(policyId).empty());
    }
}

TEST_F(IntegrationHeuristicPluginLoadedNoOptional, PluginWithoutOptionalSetLogLevelSucceeds)
{
    // Plugin doesn't implement hipdnnHeuristicSetLogLevel
    // Should return SUCCESS without calling the function
    const auto status = plugin().setLogLevel(HIPDNN_SEV_INFO);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
}

TEST_F(IntegrationHeuristicPluginLoadedNoOptional, PluginWithoutOptionalCanStillExecuteWorkflow)
{
    // Full workflow should work despite missing optional functions
    const auto handleGuard = makeScopedHandle(plugin(), plugin().createHandle());
    ASSERT_NE(handleGuard.get(), nullptr);

    const auto policyId = plugin().getAllPolicyIds().front();
    const auto descGuard = makeScopedPolicyDescriptor(
        plugin(), plugin().createPolicyDescriptor(handleGuard.get(), policyId));
    ASSERT_NE(descGuard.get(), nullptr);

    const std::vector<int64_t> inputIds = {1, 2, 3};
    plugin().setEngineIds(descGuard.get(), inputIds.data(), inputIds.size());

    const bool applied = plugin().finalize(descGuard.get());
    EXPECT_FALSE(applied); // This plugin declines to apply

    const auto sortedIds = plugin().getSortedEngineIds(descGuard.get());
    EXPECT_TRUE(sortedIds.empty()); // Returns empty list
}

// ====================================================================================
// IntegrationHeuristicPluginIncomplete: plugin missing required symbols is rejected
// ====================================================================================
//
// This fixture cannot pre-load the plugin in SetUp because construction is expected
// to throw. Each test loads the SharedLibrary and attempts construction in its body.

class IntegrationHeuristicPluginIncomplete : public ::testing::Test
{
protected:
    static std::filesystem::path incompletePluginPath()
    {
        return getHeuristicPluginPath(TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME);
    }
};

TEST_F(IntegrationHeuristicPluginIncomplete, LoadIncompletePluginThrowsException)
{
    const auto pluginPath = incompletePluginPath();

    ASSERT_TRUE(std::filesystem::exists(pluginPath)) << "Test plugin not found: " << pluginPath;

    SharedLibrary lib(pluginPath);

    // Loading should fail during symbol resolution
    EXPECT_THROW(
        {
            try
            {
                const TestableHeuristicPlugin plugin(std::move(lib));
            }
            catch(const HipdnnException& e)
            {
                // Verify the exception contains expected error details
                const std::string errorMsg(e.what());
                EXPECT_NE(errorMsg.find("HEURISTIC PLUGIN ABI INCOMPLETE"), std::string::npos);
                EXPECT_NE(errorMsg.find("Missing required symbol"), std::string::npos);
                // Error text uses SharedLibrary's weakly_canonical path; on Windows the string can
                // differ in drive-letter case or separators from a fresh weakly_canonical(pluginPath).
                const auto canonicalPath = std::filesystem::weakly_canonical(pluginPath);
                static constexpr std::string_view K_PLUGIN_PREFIX{"Plugin: "};
                const auto prefixPos = errorMsg.find(K_PLUGIN_PREFIX);
                ASSERT_NE(prefixPos, std::string::npos);
                const auto pathStart = prefixPos + K_PLUGIN_PREFIX.size();
                const auto pathEnd = errorMsg.find('\n', pathStart);
                ASSERT_NE(pathEnd, std::string::npos);
                const std::filesystem::path pluginPathInMessage(
                    errorMsg.substr(pathStart, pathEnd - pathStart));
                EXPECT_TRUE(
                    hipdnn_data_sdk::utilities::pathCompEq(pluginPathInMessage, canonicalPath))
                    << "pluginPathInMessage='" << pluginPathInMessage.string()
                    << "' canonicalPath='" << canonicalPath.string() << "'";
                throw;
            }
        },
        HipdnnException);
}

TEST_F(IntegrationHeuristicPluginIncomplete, IncompletePluginExceptionContainsSymbolName)
{
    const auto pluginPath = incompletePluginPath();
    SharedLibrary lib(pluginPath);

    EXPECT_THROW(
        {
            try
            {
                const TestableHeuristicPlugin plugin(std::move(lib));
            }
            catch(const HipdnnException& e)
            {
                const std::string errorMsg(e.what());
                // Should mention one of the missing required symbols
                const bool hasPluginNameError
                    = errorMsg.find("hipdnnPluginGetName") != std::string::npos;
                const bool hasFinalizeError
                    = errorMsg.find("hipdnnHeuristicPolicyFinalize") != std::string::npos;
                const bool hasGetSortedError
                    = errorMsg.find("hipdnnHeuristicPolicyGetSortedEngineIds") != std::string::npos;
                EXPECT_TRUE(hasPluginNameError || hasFinalizeError || hasGetSortedError);
                throw;
            }
        },
        HipdnnException);
}

TEST_F(IntegrationHeuristicPluginIncomplete, IncompletePluginExceptionHasPluginErrorStatus)
{
    const auto pluginPath = incompletePluginPath();
    SharedLibrary lib(pluginPath);

    EXPECT_THROW(
        {
            try
            {
                const TestableHeuristicPlugin plugin(std::move(lib));
            }
            catch(const HipdnnException& e)
            {
                EXPECT_EQ(e.getStatus(), HIPDNN_STATUS_PLUGIN_ERROR);
                throw;
            }
        },
        HipdnnException);
}
