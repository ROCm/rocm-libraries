// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <iostream>
#include <set>

#include "harness/SharedHandle.hpp"
#include "harness/TestConfig.hpp"

namespace {

std::set<std::string> getLoadedPluginNames(hipdnnHandle_t handle) {
    size_t numEngines = 0;
    if (hipdnnGetEngineCount_ext(handle, &numEngines) != HIPDNN_STATUS_SUCCESS || numEngines == 0) {
        return {};
    }

    std::set<std::string> pluginNames;
    for (size_t i = 0; i < numEngines; ++i) {
        // Two-call pattern: query required buffer sizes
        size_t engineNameLen = 0, pluginNameLen = 0, versionLen = 0, typeLen = 0;
        int64_t engineId = 0;
        hipdnnGetEngineInfo_ext(handle, i, &engineId, nullptr, &engineNameLen, nullptr,
                                &pluginNameLen, nullptr, &versionLen, nullptr, &typeLen);

        // All four buffers required on second call
        std::string engineName(engineNameLen, '\0');
        std::string pluginName(pluginNameLen, '\0');
        std::string version(versionLen, '\0');
        std::string type(typeLen, '\0');
        hipdnnGetEngineInfo_ext(handle, i, &engineId, engineName.data(), &engineNameLen,
                                pluginName.data(), &pluginNameLen, version.data(), &versionLen,
                                type.data(), &typeLen);

        // Trim null terminator
        if (!pluginName.empty() && pluginName.back() == '\0') {
            pluginName.pop_back();
        }
        pluginNames.insert(pluginName);
    }
    return pluginNames;
}

std::string formatPluginSet(const std::set<std::string>& plugins) {
    std::string result = "[";
    bool first = true;
    for (const auto& plugin : plugins) {
        if (!first) {
            result += ", ";
        } else {
            first = false;
        }
        result += plugin;
    }
    result += "]";
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize test logging infrastructure to forward logs to std::cerr based
    // on the current environment HIPDNN_LOG_LEVEL value when this function is called.
    auto recordingCallback = hipdnn_test_sdk::utilities::initializeTestLogRecordingShared();

    // Initialize plugin logger with test recording callback so that plugin logs
    // are routed to the log recorder for capture.
    hipdnn_plugin_sdk::logging::initializeCallbackLogging("hipdnn_integration_tests",
                                                          recordingCallback);

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    // Set stream on shared handle
    auto handle = hipdnn_integration_tests::getSharedHandle();
    hipStream_t stream;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::cerr << "Failed to create HIP stream" << std::endl;
        return 1;
    }
    if (hipdnnSetStream(handle, stream) != HIPDNN_STATUS_SUCCESS) {
        std::cerr << "Failed to set stream on shared handle" << std::endl;
        return 1;
    }

    // Verify loaded plugins match expected plugins
    auto expectedPlugins = hipdnn_integration_tests::TestConfig::get().getExpectedPluginNames();
    auto loadedPlugins = getLoadedPluginNames(handle);
    if (expectedPlugins != loadedPlugins) {
        std::cerr << "Plugin mismatch!\n"
                  << "  Expected: " << formatPluginSet(expectedPlugins) << "\n"
                  << "  Loaded:   " << formatPluginSet(loadedPlugins) << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    // Clean up shared handle and stream
    hipStreamDestroy(stream);
    hipdnnDestroy(handle);
    return result;
}
