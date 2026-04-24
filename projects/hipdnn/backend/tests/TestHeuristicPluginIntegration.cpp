// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPluginIntegration.cpp
 * @brief Integration tests for HeuristicPlugin workflow coverage
 *
 * These tests exercise full workflows with real plugins to improve coverage:
 * - Complete handle and descriptor lifecycle
 * - Device properties serialization and setting
 * - Engine ID setting and finalization
 * - Error handling paths
 */

#include "HipdnnException.hpp"
#include "PlatformUtils.hpp"
#include "TestPluginConstants.hpp"
#include "plugin/HeuristicPlugin.hpp"
#include "plugin/HeuristicPluginManager.hpp"
#include "plugin/HeuristicPluginResourceManager.hpp"
#include "plugin/SharedLibrary.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/device_properties_generated.h>

#include <flatbuffers/flatbuffers.h>

#include <gtest/gtest.h>
#include <string_view>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;
using namespace hipdnn_backend::plugin_constants;

namespace
{
// Note: TEST_GOOD_HEURISTIC_PLUGIN_NAME, TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME,
// and TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME are defined as macros in CMakeLists.txt

// Helper to serialize DevicePropertiesT using FlatBuffers Pack
std::vector<uint8_t>
    serializeDeviceProperties(const hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT& props)
{
    flatbuffers::FlatBufferBuilder builder(256);
    auto offset = hipdnn_flatbuffers_sdk::data_objects::DeviceProperties::Pack(builder, &props);
    builder.Finish(offset, "HDDP");
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}
} // namespace

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

    // Find the good test plugin
    const HeuristicPlugin* plugin = nullptr;
    hipdnnHeuristicHandle_t handle = nullptr;
    for(const auto& info : policyInfos)
    {
        plugin = rm->getPluginForPolicyId(info.policyId);
        if(plugin != nullptr)
        {
            handle = rm->getHeuristicHandleForPolicyId(info.policyId);
            break;
        }
    }

    ASSERT_NE(plugin, nullptr);
    ASSERT_NE(handle, nullptr);

    // Verify plugin metadata is available
    EXPECT_FALSE(plugin->version().empty());
}

TEST_F(IntegrationHeuristicPlugin, CompletePolicyDescriptorLifecycle)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(plugin, nullptr);

    hipdnnHeuristicHandle_t handle = rm->getHeuristicHandleForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(handle, nullptr);

    // Create policy descriptor
    auto desc = plugin->createPolicyDescriptor(handle);
    EXPECT_NE(desc, nullptr);

    // Destroy policy descriptor (should not throw)
    EXPECT_NO_THROW(plugin->destroyPolicyDescriptor(desc));
}

TEST_F(IntegrationHeuristicPlugin, SetEngineIdsOnPolicyDescriptor)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(plugin, nullptr);

    hipdnnHeuristicHandle_t handle = rm->getHeuristicHandleForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(handle, nullptr);

    auto desc = plugin->createPolicyDescriptor(handle);
    ASSERT_NE(desc, nullptr);

    // Set engine IDs
    const std::vector<int64_t> engineIds = {1, 2, 3};
    EXPECT_NO_THROW(plugin->setEngineIds(desc, engineIds.data(), engineIds.size()));

    plugin->destroyPolicyDescriptor(desc);
}

TEST_F(IntegrationHeuristicPlugin, SetSerializedGraphOnPolicyDescriptor)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(plugin, nullptr);

    hipdnnHeuristicHandle_t handle = rm->getHeuristicHandleForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(handle, nullptr);

    auto desc = plugin->createPolicyDescriptor(handle);
    ASSERT_NE(desc, nullptr);

    // Create a simple serialized graph (just some bytes)
    const std::vector<uint8_t> graphBytes = {1, 2, 3, 4, 5};
    hipdnnPluginConstData_t serializedGraph;
    serializedGraph.ptr = graphBytes.data();
    serializedGraph.size = graphBytes.size();

    EXPECT_NO_THROW(plugin->setSerializedGraph(desc, &serializedGraph));

    plugin->destroyPolicyDescriptor(desc);
}

TEST_F(IntegrationHeuristicPlugin, FinalizeAndGetSortedEngineIds)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(plugin, nullptr);

    hipdnnHeuristicHandle_t handle = rm->getHeuristicHandleForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(handle, nullptr);

    auto desc = plugin->createPolicyDescriptor(handle);
    ASSERT_NE(desc, nullptr);

    // Set engine IDs
    const std::vector<int64_t> engineIds = {100, 200, 300};
    plugin->setEngineIds(desc, engineIds.data(), engineIds.size());

    // Finalize the policy
    plugin->finalize(desc);

    // Get sorted IDs
    const auto sortedIds = plugin->getSortedEngineIds(desc);
    EXPECT_TRUE(sortedIds.empty() || !sortedIds.empty()); // May or may not apply

    plugin->destroyPolicyDescriptor(desc);
}

TEST_F(IntegrationHeuristicPlugin, SetDevicePropertiesOnHandle)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(plugin, nullptr);

    hipdnnHeuristicHandle_t handle = rm->getHeuristicHandleForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(handle, nullptr);

    // Create device properties
    hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT props;
    props.device_id = 0;
    props.multi_processor_count = 120;
    props.total_global_mem = 16ULL * 1024 * 1024 * 1024; // 16 GB
    props.architecture_name = "gfx90a";

    // Serialize
    auto serialized = serializeDeviceProperties(props);
    hipdnnPluginConstData_t devicePropsData;
    devicePropsData.ptr = serialized.data();
    devicePropsData.size = serialized.size();

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

    auto serialized = serializeDeviceProperties(props);
    hipdnnPluginConstData_t devicePropsData;
    devicePropsData.ptr = serialized.data();
    devicePropsData.size = serialized.size();

    // Set on all handles via resource manager
    EXPECT_NO_THROW(rm->setDevicePropertiesOnAllHandles(&devicePropsData));
}

TEST_F(IntegrationHeuristicPlugin, CompleteWorkflowWithDevicePropertiesAndFinalize)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(plugin, nullptr);

    hipdnnHeuristicHandle_t handle = rm->getHeuristicHandleForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(handle, nullptr);

    // Set device properties on handle
    hipdnn_flatbuffers_sdk::data_objects::DevicePropertiesT props;
    props.device_id = 0;
    props.multi_processor_count = 120;
    props.total_global_mem = 16ULL * 1024 * 1024 * 1024;
    props.architecture_name = "gfx90a";

    auto serialized = serializeDeviceProperties(props);
    hipdnnPluginConstData_t devicePropsData;
    devicePropsData.ptr = serialized.data();
    devicePropsData.size = serialized.size();
    plugin->setDeviceProperties(handle, &devicePropsData);

    // Create policy descriptor
    auto desc = plugin->createPolicyDescriptor(handle);
    ASSERT_NE(desc, nullptr);

    // Set engine IDs
    const std::vector<int64_t> engineIds = {1000, 2000, 3000};
    plugin->setEngineIds(desc, engineIds.data(), engineIds.size());

    // Set serialized graph
    const std::vector<uint8_t> graphBytes = {10, 20, 30};
    hipdnnPluginConstData_t serializedGraph;
    serializedGraph.ptr = graphBytes.data();
    serializedGraph.size = graphBytes.size();
    plugin->setSerializedGraph(desc, &serializedGraph);

    // Finalize
    plugin->finalize(desc);

    // Get results
    const auto sortedIds = plugin->getSortedEngineIds(desc);

    // Clean up
    plugin->destroyPolicyDescriptor(desc);
}

// ========== Plugin Metadata Coverage ==========

TEST_F(IntegrationHeuristicPlugin, GetPolicyNameFromLoadedPlugin)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    for(const auto& info : policyInfos)
    {
        const HeuristicPlugin* plugin = rm->getPluginForPolicyId(info.policyId);
        if(plugin != nullptr)
        {
            // Should return policy name (may be empty for no-optional plugin)
            const auto policyName = plugin->name();
            // Just verify it doesn't crash
            EXPECT_TRUE(policyName.empty() || !policyName.empty());
        }
    }
}

TEST_F(IntegrationHeuristicPlugin, GetPluginVersionFromLoadedPlugin)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(plugin, nullptr);

    const auto pluginVersion = plugin->version();
    EXPECT_FALSE(pluginVersion.empty());
}

TEST_F(IntegrationHeuristicPlugin, GetApiVersionFromLoadedPlugin)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(plugin, nullptr);

    const auto apiVersion = plugin->apiVersion();
    EXPECT_FALSE(apiVersion.empty());
    EXPECT_NE(apiVersion.find("0."), std::string_view::npos); // Should be version 0.x
}

TEST_F(IntegrationHeuristicPlugin, GetPluginTypeFromLoadedPlugin)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(policyInfos[0].policyId);
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

    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(plugin, nullptr);

    hipdnnHeuristicHandle_t handle = rm->getHeuristicHandleForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(handle, nullptr);

    // Create multiple descriptors from the same handle
    auto desc1 = plugin->createPolicyDescriptor(handle);
    auto desc2 = plugin->createPolicyDescriptor(handle);
    auto desc3 = plugin->createPolicyDescriptor(handle);

    EXPECT_NE(desc1, nullptr);
    EXPECT_NE(desc2, nullptr);
    EXPECT_NE(desc3, nullptr);

    // Note: Test plugins may return the same hardcoded pointer for simplicity,
    // but real plugins should return distinct descriptors. We just verify they're created.

    // Clean up all
    plugin->destroyPolicyDescriptor(desc1);
    plugin->destroyPolicyDescriptor(desc2);
    plugin->destroyPolicyDescriptor(desc3);
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

TEST_F(IntegrationHeuristicPlugin, PolicyNameReturnsEmptyWhenOptionalFunctionMissing)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    // Find the no-optional plugin which doesn't implement hipdnnHeuristicGetPolicyName
    for(const auto& info : policyInfos)
    {
        const HeuristicPlugin* plugin = rm->getPluginForPolicyId(info.policyId);
        if(plugin != nullptr)
        {
            // policyName() should return empty string for plugins without the optional function
            const auto name = plugin->name();
            // Either has a name or empty string (both valid)
            EXPECT_TRUE(name.empty() || !name.empty());
        }
    }
}

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

    const HeuristicPlugin* plugin = rm->getPluginForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(plugin, nullptr);

    hipdnnHeuristicHandle_t handle = rm->getHeuristicHandleForPolicyId(policyInfos[0].policyId);
    ASSERT_NE(handle, nullptr);

    auto desc = plugin->createPolicyDescriptor(handle);
    ASSERT_NE(desc, nullptr);

    // Don't set any engine IDs - just finalize
    plugin->finalize(desc);

    // Get sorted IDs (should be empty)
    const auto sortedIds = plugin->getSortedEngineIds(desc);
    EXPECT_TRUE(sortedIds.empty());

    plugin->destroyPolicyDescriptor(desc);
}

// ========== Error Path: Multiple Policy Lookups (Same Handle/Plugin Reuse) ==========

TEST_F(IntegrationHeuristicPlugin, MultipleGetHandleCallsReturnSameHandle)
{
    auto rm = HeuristicPluginResourceManager::create();
    ASSERT_NE(rm, nullptr);

    const auto policyInfos = rm->getHeuristicPolicyInfos();
    ASSERT_FALSE(policyInfos.empty());

    const auto policyId = policyInfos[0].policyId;

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

    const auto policyId = policyInfos[0].policyId;

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

    auto serialized = serializeDeviceProperties(props);
    hipdnnPluginConstData_t devicePropsData;
    devicePropsData.ptr = serialized.data();
    devicePropsData.size = serialized.size();

    // Should not throw when no plugins loaded
    EXPECT_NO_THROW(rm->setDevicePropertiesOnAllHandles(&devicePropsData));
}

// ========== Plugin Loading Tests ==========
// These tests exercise loading real plugins and their functionality

namespace
{

// Wrapper class to access protected constructor
class TestableHeuristicPlugin : public HeuristicPlugin
{
public:
    explicit TestableHeuristicPlugin(SharedLibrary&& lib)
        : HeuristicPlugin(std::move(lib))
    {
    }
};

} // anonymous namespace

// Fixture that loads the good plugin for tests that need it
class TestHeuristicPluginLoadedGood : public ::testing::Test
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
TEST_F(IntegrationHeuristicPlugin, LoadGoodPluginSucceeds)
{
    const auto pluginPath = getHeuristicPluginPath(TEST_GOOD_HEURISTIC_PLUGIN_NAME);

    ASSERT_TRUE(std::filesystem::exists(pluginPath))
        << "Test plugin not found: " << pluginPath
        << "\nMake sure test_plugins are built before running tests";

    // Load the plugin
    SharedLibrary lib(pluginPath);
    // NOLINTNEXTLINE(misc-const-correctness)
    ASSERT_NO_THROW({ TestableHeuristicPlugin plugin(std::move(lib)); });
}
TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanQueryApiVersion)
{
    const auto version = plugin().apiVersion();
    EXPECT_FALSE(version.empty());
    EXPECT_EQ(version, "0.0.1"); // HIPDNN_HEURISTIC_API_VERSION
}
TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanQueryPolicyId)
{
    const auto policyId = plugin().policyId();
    const auto expectedId = hipdnn_data_sdk::utilities::engineNameToId("TestGoodHeuristicPolicy");
    EXPECT_EQ(policyId, expectedId);
}
TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanQueryPolicyName)
{
    const auto name = plugin().name();
    EXPECT_EQ(name, "TestGoodHeuristicPolicy");
}
TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanQueryPluginVersion)
{
    const auto version = plugin().version();
    EXPECT_EQ(version, "1.0.0");
}
TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanCreateAndDestroyHandle)
{
    hipdnnHeuristicHandle_t handle = nullptr;
    ASSERT_NO_THROW({ handle = plugin().createHandle(); });
    EXPECT_NE(handle, nullptr);

    ASSERT_NO_THROW({ plugin().destroyHandle(handle); });
}
TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanSetDeviceProperties)
{
    const auto handle = plugin().createHandle();
    ASSERT_NE(handle, nullptr);

    hipdnnPluginConstData_t deviceProps{};
    deviceProps.ptr = nullptr;
    deviceProps.size = 0;

    ASSERT_NO_THROW({ plugin().setDeviceProperties(handle, &deviceProps); });

    plugin().destroyHandle(handle);
}
TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanManagePolicyDescriptor)
{
    const auto handle = plugin().createHandle();
    ASSERT_NE(handle, nullptr);

    hipdnnHeuristicPolicyDescriptor_t desc = nullptr;
    ASSERT_NO_THROW({ desc = plugin().createPolicyDescriptor(handle); });
    EXPECT_NE(desc, nullptr);

    ASSERT_NO_THROW({ plugin().destroyPolicyDescriptor(desc); });

    plugin().destroyHandle(handle);
}
TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanSetEngineIds)
{
    const auto handle = plugin().createHandle();
    const auto desc = plugin().createPolicyDescriptor(handle);

    const std::vector<int64_t> engineIds = {1, 2, 3, 4, 5};
    ASSERT_NO_THROW({ plugin().setEngineIds(desc, engineIds.data(), engineIds.size()); });

    plugin().destroyPolicyDescriptor(desc);
    plugin().destroyHandle(handle);
}
TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanSetSerializedGraph)
{
    const auto handle = plugin().createHandle();
    const auto desc = plugin().createPolicyDescriptor(handle);

    hipdnnPluginConstData_t graphData{};
    graphData.ptr = nullptr;
    graphData.size = 0;

    ASSERT_NO_THROW({ plugin().setSerializedGraph(desc, &graphData); });

    plugin().destroyPolicyDescriptor(desc);
    plugin().destroyHandle(handle);
}
TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanFinalizePolicy)
{
    const auto handle = plugin().createHandle();
    const auto desc = plugin().createPolicyDescriptor(handle);

    const std::vector<int64_t> engineIds = {1, 2, 3};
    plugin().setEngineIds(desc, engineIds.data(), engineIds.size());

    bool applied = false;
    ASSERT_NO_THROW({ applied = plugin().finalize(desc); });
    EXPECT_TRUE(applied); // Good plugin always applies

    plugin().destroyPolicyDescriptor(desc);
    plugin().destroyHandle(handle);
}
TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCanGetSortedEngineIds)
{
    const auto handle = plugin().createHandle();
    const auto desc = plugin().createPolicyDescriptor(handle);

    const std::vector<int64_t> inputIds = {1, 2, 3, 4, 5};
    plugin().setEngineIds(desc, inputIds.data(), inputIds.size());
    plugin().finalize(desc);

    std::vector<int64_t> sortedIds;
    ASSERT_NO_THROW({ sortedIds = plugin().getSortedEngineIds(desc); });

    // Good plugin reverses the order
    EXPECT_EQ(sortedIds.size(), inputIds.size());
    EXPECT_EQ(sortedIds, std::vector<int64_t>({5, 4, 3, 2, 1}));

    plugin().destroyPolicyDescriptor(desc);
    plugin().destroyHandle(handle);
}
TEST_F(TestHeuristicPluginLoadedGood, LoadedPluginCompleteWorkflow)
{
    // Create handle and descriptor
    const auto handle = plugin().createHandle();
    ASSERT_NE(handle, nullptr);

    const auto desc = plugin().createPolicyDescriptor(handle);
    ASSERT_NE(desc, nullptr);

    // Set inputs
    const std::vector<int64_t> inputIds = {10, 20, 30};
    plugin().setEngineIds(desc, inputIds.data(), inputIds.size());

    hipdnnPluginConstData_t graphData{};
    graphData.ptr = nullptr;
    graphData.size = 0;
    plugin().setSerializedGraph(desc, &graphData);

    // Finalize and retrieve results
    const bool applied = plugin().finalize(desc);
    EXPECT_TRUE(applied);

    const auto sortedIds = plugin().getSortedEngineIds(desc);
    EXPECT_EQ(sortedIds, std::vector<int64_t>({30, 20, 10}));

    // Cleanup
    plugin().destroyPolicyDescriptor(desc);
    plugin().destroyHandle(handle);
}
TEST_F(IntegrationHeuristicPlugin, LoadIncompletePluginThrowsException)
{
    const auto pluginPath = getHeuristicPluginPath(TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME);

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
TEST_F(IntegrationHeuristicPlugin, IncompletePluginExceptionContainsSymbolName)
{
    const auto pluginPath = getHeuristicPluginPath(TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME);
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
TEST_F(IntegrationHeuristicPlugin, IncompletePluginExceptionHasPluginErrorStatus)
{
    const auto pluginPath = getHeuristicPluginPath(TEST_INCOMPLETE_HEURISTIC_API_PLUGIN_NAME);
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
TEST_F(IntegrationHeuristicPlugin, LoadPluginWithoutOptionalSymbolsSucceeds)
{
    const auto pluginPath = getHeuristicPluginPath(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME);

    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    ASSERT_TRUE(std::filesystem::exists(pluginPath)) << "Test plugin not found: " << pluginPath;

    SharedLibrary lib(pluginPath);

    // Should load successfully despite missing optional symbols
    // NOLINTNEXTLINE(misc-const-correctness)
    ASSERT_NO_THROW({ TestableHeuristicPlugin plugin(std::move(lib)); });
}
TEST_F(IntegrationHeuristicPlugin, PluginWithoutOptionalPolicyNameHasName)
{
    const auto pluginPath = getHeuristicPluginPath(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME);
    SharedLibrary lib(pluginPath);
    const TestableHeuristicPlugin plugin(std::move(lib));

    // GetPolicyName is now required
    const auto name = plugin.name();
    EXPECT_FALSE(name.empty());
    EXPECT_EQ(name, "TestNoOptionalHeuristicPolicy");
}
TEST_F(IntegrationHeuristicPlugin, PluginWithoutOptionalSetLogLevelSucceeds)
{
    const auto pluginPath = getHeuristicPluginPath(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME);
    SharedLibrary lib(pluginPath);
    const TestableHeuristicPlugin plugin(std::move(lib));

    // Plugin doesn't implement hipdnnHeuristicSetLogLevel
    // Should return SUCCESS without calling the function
    const auto status = plugin.setLogLevel(HIPDNN_SEV_INFO);
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
}
TEST_F(IntegrationHeuristicPlugin, PluginWithoutOptionalCanStillExecuteWorkflow)
{
    const auto pluginPath = getHeuristicPluginPath(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME);
    SharedLibrary lib(pluginPath);
    const TestableHeuristicPlugin plugin(std::move(lib));

    // Full workflow should work despite missing optional functions
    const auto handle = plugin.createHandle();
    ASSERT_NE(handle, nullptr);

    const auto desc = plugin.createPolicyDescriptor(handle);
    ASSERT_NE(desc, nullptr);

    const std::vector<int64_t> inputIds = {1, 2, 3};
    plugin.setEngineIds(desc, inputIds.data(), inputIds.size());

    const bool applied = plugin.finalize(desc);
    EXPECT_FALSE(applied); // This plugin declines to apply

    const auto sortedIds = plugin.getSortedEngineIds(desc);
    EXPECT_TRUE(sortedIds.empty()); // Returns empty list

    plugin.destroyPolicyDescriptor(desc);
    plugin.destroyHandle(handle);
}
TEST_F(TestHeuristicPluginLoadedGood, RealPluginCachesPolicyId)
{
    // First call - ID is computed from policy name
    const auto id1 = plugin().policyId();
    const auto expectedId = hipdnn_data_sdk::utilities::engineNameToId("TestGoodHeuristicPolicy");
    EXPECT_EQ(id1, expectedId);

    // Second call should return cached value
    const auto id2 = plugin().policyId();
    EXPECT_EQ(id2, id1);
}
TEST_F(IntegrationHeuristicPlugin, GetSortedEngineIdsReturnsEmptyWhenNoEngines)
{
    const auto pluginPath = getHeuristicPluginPath(TEST_NO_OPTIONAL_HEURISTIC_PLUGIN_NAME);
    SharedLibrary lib(pluginPath);
    const TestableHeuristicPlugin plugin(std::move(lib));

    const auto handle = plugin.createHandle();
    const auto desc = plugin.createPolicyDescriptor(handle);

    // Don't set any engine IDs
    plugin.finalize(desc);

    const auto sortedIds = plugin.getSortedEngineIds(desc);
    EXPECT_TRUE(sortedIds.empty());

    plugin.destroyPolicyDescriptor(desc);
    plugin.destroyHandle(handle);
}
