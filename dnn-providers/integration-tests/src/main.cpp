// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <iostream>
#include <optional>
#include <string>

#include "harness/SharedHandle.hpp"
#include "harness/TestConfig.hpp"

namespace
{

// Extract and remove a CLI argument from argv
std::optional<std::string> extractArg(int& argc, char** argv, std::string_view flag)
{
    for(int i = 1; i < argc; ++i)
    {
        if(argv[i] == flag)
        {
            if(i + 1 >= argc)
            {
                return std::nullopt; // flag present but no value
            }
            std::optional<std::string> value = argv[i + 1];
            // Shift arguments to remove the flag and its value
            for(int j = i; j < argc - 2; ++j)
            {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            return value;
        }
    }
    return std::nullopt;
}

void printUsage(const char* programName)
{
    std::cerr << "Usage: " << programName
              << " --test-article <path> --test-engine <name> [gtest options]\n"
              << "  --test-article <path>  Full path to the hipdnn engine plugin .so to test\n"
              << "  --test-engine <name>   Engine name to test against (e.g., MIOPEN_ENGINE)\n"
              << "  [gtest options]        Standard gtest options (--gtest_filter, etc.)\n";
}

bool engineIsLoaded(hipdnnHandle_t handle, std::string_view targetEngineName)
{
    size_t numEngines = 0;
    if(hipdnnGetEngineCount_ext(handle, &numEngines) != HIPDNN_STATUS_SUCCESS || numEngines == 0)
    {
        return false;
    }

    for(size_t i = 0; i < numEngines; ++i)
    {
        // Two-call pattern: first call queries required buffer sizes
        size_t engineNameLen = 0;
        size_t pluginNameLen = 0;
        size_t versionLen = 0;
        size_t typeLen = 0;
        int64_t engineId = 0;
        hipdnnGetEngineInfo_ext(handle,
                                i,
                                &engineId,
                                nullptr,
                                &engineNameLen,
                                nullptr,
                                &pluginNameLen,
                                nullptr,
                                &versionLen,
                                nullptr,
                                &typeLen);

        // Second call: ALL four string buffers must be non-null
        std::string engineName(engineNameLen, '\0');
        std::string pluginName(pluginNameLen, '\0');
        std::string version(versionLen, '\0');
        std::string type(typeLen, '\0');
        hipdnnGetEngineInfo_ext(handle,
                                i,
                                &engineId,
                                engineName.data(),
                                &engineNameLen,
                                pluginName.data(),
                                &pluginNameLen,
                                version.data(),
                                &versionLen,
                                type.data(),
                                &typeLen);

        // Trim null terminator
        if(!engineName.empty() && engineName.back() == '\0')
        {
            engineName.pop_back();
        }

        if(engineName == targetEngineName)
        {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char** argv) noexcept
{
    try
    {
        // Parse custom arguments before InitGoogleTest to avoid unknown flag warnings
        auto articlePath = extractArg(argc, argv, "--test-article");
        auto engineName = extractArg(argc, argv, "--test-engine");

        if(!articlePath || !engineName)
        {
            printUsage(argv[0]);
            return 1;
        }

        // Validate and canonicalize article path (resolves relative paths)
        std::filesystem::path articlePathObj;
        try
        {
            articlePathObj = std::filesystem::canonical(*articlePath);
        }
        catch(const std::filesystem::filesystem_error&)
        {
            std::cerr << "Error: Article path does not exist: " << *articlePath << '\n';
            return 1;
        }

        // Set engine plugin path to the plugin file (not the directory)
        const std::string articlePathStr = articlePathObj.string();
        const char* pluginPath = articlePathStr.c_str();
        if(hipdnnSetEnginePluginPaths_ext(1, &pluginPath, HIPDNN_PLUGIN_LOADING_ABSOLUTE)
           != HIPDNN_STATUS_SUCCESS)
        {
            std::cerr << "Error: Failed to set engine plugin path\n";
            return 1;
        }

        // Initialize TestConfig with CLI arguments
        hipdnn_integration_tests::TestConfig::initialize(std::move(articlePathObj),
                                                         std::move(*engineName));

        // Now safe to call InitGoogleTest (all custom flags removed)
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

        // Create shared handle (triggers engine loading)
        auto handle = hipdnn_integration_tests::getSharedHandle();

        // Set stream on shared handle
        hipStream_t stream;
        if(hipStreamCreate(&stream) != hipSuccess)
        {
            std::cerr << "Failed to create HIP stream\n";
            return 1;
        }
        if(hipdnnSetStream(handle, stream) != HIPDNN_STATUS_SUCCESS)
        {
            std::cerr << "Failed to set stream on shared handle\n";
            return 1;
        }

        // Verify target engine is loaded
        if(!engineIsLoaded(handle, hipdnn_integration_tests::TestConfig::get().getEngineName()))
        {
            std::cerr << "Error: Engine '"
                      << hipdnn_integration_tests::TestConfig::get().getEngineName()
                      << "' is not loaded. Check the plugin path.\n";
            return 1;
        }

        const int result = RUN_ALL_TESTS();

        // Clean up shared handle and stream
        static_cast<void>(hipStreamDestroy(stream));
        hipdnnDestroy(handle);
        return result;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}
