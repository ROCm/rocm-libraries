// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/ScopedResource.hpp>

#include "plugin/EnginePlugin.hpp"

using namespace hipdnn_backend;
using namespace hipdnn_sdk::utilities;

class SimpleEnginePluginManager : public plugin::PluginManagerBase<plugin::EnginePlugin>
{
public:
    SimpleEnginePluginManager()
        : plugin::PluginManagerBase<plugin::EnginePlugin>({})
    {
    }
};

TEST(TestGpuEnginePluginManager, LoadPluginsAndExecuteOpGraph)
{
    SKIP_IF_NO_DEVICES();

    // Create an SimpleEnginePluginManager instance
    SimpleEnginePluginManager pluginManager;

    // Create a list of paths to plugins
    std::set<std::filesystem::path> pluginPaths = {"../lib/test_plugins/" TEST_ENGINE_PLUGIN1_NAME};

    // Load the plugins
    pluginManager.loadPlugins(pluginPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);

    const auto& plugins = pluginManager.getPlugins();
    ASSERT_EQ(plugins.size(), 1); // Ensure one plugin is loaded

    // Check that the plugins have the correct names
    ASSERT_EQ(plugins[0]->name(), "EnginePlugin1");

    // Check that the plugins have the correct versions
    ASSERT_EQ(plugins[0]->version(), "1.0");

    // Check all engine IDs
    std::vector<int64_t> expectedEngineIds0 = {100, 101, 102};
    const auto& engineIds0 = plugins[0]->getAllEngineIds();
    ASSERT_EQ(engineIds0, expectedEngineIds0);

    auto stream = reinterpret_cast<hipStream_t>(0x1234);

    // TODO set a real op graph
    const std::array<uint8_t, 8> opGraphData = {0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00};
    const hipdnnPluginConstData_t opGraph = {opGraphData.data(), opGraphData.size()};

    // The number of elements in the input vector.
    const unsigned dataSize = 512;

    // The number of bytes to allocate for the input- and output device vectors.
    const std::size_t sizeBytes = dataSize * sizeof(uint32_t);

    // Allocate host input buffer and fill it with an increasing sequence (i.e. 0, 1, 2, ...).
    std::vector<uint32_t> inHostData(dataSize);
    std::iota(inHostData.begin(), inHostData.end(), 0);

    // Host output buffer.
    std::vector<uint32_t> outHostData(dataSize);

    // Allocate input device buffer.
    uint32_t* inDevData{};
    ASSERT_EQ(hipMalloc(&inDevData, sizeBytes), hipSuccess);
    ScopedResource inDevDataRes(inDevData, [](uint32_t* ptr) { std::ignore = hipFree(ptr); });

    // Allocate output device buffer.
    uint32_t* outDevData{};
    ASSERT_EQ(hipMalloc(&outDevData, sizeBytes), hipSuccess);
    ScopedResource outDevDataRes(outDevData, [](uint32_t* ptr) { std::ignore = hipFree(ptr); });

    // Copy the input data from the host to the device.
    ASSERT_EQ(hipMemcpy(inDevData, inHostData.data(), sizeBytes, hipMemcpyHostToDevice),
              hipSuccess);

    for(const auto& plugin : plugins)
    {
        auto handle = plugin->createHandle();
        ScopedResource handleRes(handle, [&plugin](auto h) { plugin->destroyHandle(h); });

        plugin->setStream(handle, stream);

        const auto engineIds = plugin->getApplicableEngineIds(handle, &opGraph);
        ASSERT_GT(engineIds.size(), 0); // Ensure at least one engine is applicable

        for(const auto engineId : engineIds)
        {
            // Get engine details
            hipdnnPluginConstData_t engineDetails;
            plugin->getEngineDetails(handle, engineId, &opGraph, &engineDetails);
            ScopedResource engineDetailsRes(&engineDetails,
                                            [handle, &plugin](hipdnnPluginConstData_t* ed) {
                                                plugin->destroyEngineDetails(handle, ed);
                                            });

            // Prepare the engine configuration
            // TODO set a real engine config based on the engine details
            const hipdnnPluginConstData_t engineConfig = {nullptr, 0};

            // Create workspace for the operation
            auto workspaceSize = plugin->getWorkspaceSize(handle, &engineConfig, &opGraph);
            void* workspace = nullptr;
            if(workspaceSize > 0)
            {
                ASSERT_EQ(hipMalloc(&workspace, workspaceSize), hipSuccess);
            }
            ScopedResource workspaceRes(workspace, [](void* ptr) { std::ignore = hipFree(ptr); });

            // Create execution context for the operation
            auto executionContext = plugin->createExecutionContext(handle, &engineConfig, &opGraph);
            ScopedResource executionContextRes(executionContext, [&plugin, handle](auto ec) {
                plugin->destroyExecutionContext(handle, ec);
            });

            // Fill output device buffer with zeros
            ASSERT_EQ(hipMemset(outDevData, 0, sizeBytes), hipSuccess);

            // Prepare device buffers structure
            const uint32_t numDeviceBuffers = 2;
            const std::array<hipdnnPluginDeviceBuffer_t, numDeviceBuffers> deviceBuffers
                = {{{0, inDevData}, {1, outDevData}}};

            // Execute the operation graph
            plugin->executeOpGraph(
                handle, executionContext, workspace, deviceBuffers.data(), numDeviceBuffers);

            // Copy the results back to the host. This call blocks the host's execution until the copy is finished.
            ASSERT_EQ(hipMemcpy(outHostData.data(), outDevData, sizeBytes, hipMemcpyDeviceToHost),
                      hipSuccess);

            // Check the results
            ASSERT_EQ(
                std::equal(
                    inHostData.begin(), inHostData.end(), outHostData.begin(), outHostData.end()),
                true);
        }
    }
}
