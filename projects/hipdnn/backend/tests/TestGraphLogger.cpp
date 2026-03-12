// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "descriptors/FlatbufferTestUtils.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "logging/GraphLogger.hpp"
#include "logging/Logging.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <nlohmann/json.hpp>
#include <thread>

using namespace hipdnn_backend;

class TestGraphLogger : public ::testing::Test
{
protected:
    std::filesystem::path _tempDir;
    std::unique_ptr<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _logLevelGuard;
    std::unique_ptr<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _logFileGuard;
    std::unique_ptr<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _logGraphGuard;

    void SetUp() override
    {
        _logLevelGuard
            = std::make_unique<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>(
                "HIPDNN_LOG_LEVEL");
        _logFileGuard
            = std::make_unique<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>(
                "HIPDNN_LOG_FILE");
        _logGraphGuard
            = std::make_unique<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>(
                "HIPDNN_LOG_GRAPH");

        hipdnn_backend::logging::loggerShutdown();

        hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "off");
        hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_FILE");
        hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_GRAPH");

        // Create a unique temp directory for each test
        _tempDir
            = std::filesystem::temp_directory_path()
              / ("hipdnn_graph_test_"
                 + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + "_"
                 + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(_tempDir);

        testing::internal::CaptureStderr();
    }

    void TearDown() override
    {
        hipdnn_backend::logging::loggerShutdown();

        testing::internal::GetCapturedStderr();

        _logGraphGuard.reset();
        _logFileGuard.reset();
        _logLevelGuard.reset();

        if(std::filesystem::exists(_tempDir))
        {
            std::filesystem::remove_all(_tempDir);
        }
    }

    static GraphDescriptor createAndFinalizeGraph()
    {
        auto builder = test_utilities::createValidGraph();
        auto serializedGraph = builder.Release();

        GraphDescriptor descriptor;
        descriptor.deserializeGraph(serializedGraph.data(), serializedGraph.size());

        auto handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
        descriptor.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
        descriptor.finalize();
        return descriptor;
    }

    static std::vector<std::filesystem::path> getJsonFilesInDir(const std::filesystem::path& dir)
    {
        std::vector<std::filesystem::path> jsonFiles;
        if(!std::filesystem::exists(dir))
        {
            return jsonFiles;
        }
        for(const auto& entry : std::filesystem::directory_iterator(dir))
        {
            if(entry.path().extension() == ".json")
            {
                jsonFiles.push_back(entry.path());
            }
        }
        return jsonFiles;
    }
};

TEST_F(TestGraphLogger, GraphNotLoggedWhenDisabled)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_GRAPH");
    auto logFilePath = (_tempDir / "hipdnn.log").string();
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", logFilePath.c_str());

    auto descriptor = createAndFinalizeGraph();

    auto jsonFiles = getJsonFilesInDir(_tempDir);
    EXPECT_TRUE(jsonFiles.empty());
}

TEST_F(TestGraphLogger, GraphNotLoggedWhenLogLevelOff)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "off");
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH", "json");
    auto logFilePath = (_tempDir / "hipdnn.log").string();
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", logFilePath.c_str());

    auto descriptor = createAndFinalizeGraph();

    auto jsonFiles = getJsonFilesInDir(_tempDir);
    EXPECT_TRUE(jsonFiles.empty());
}

TEST_F(TestGraphLogger, GraphNotLoggedWhenLogLevelWarn)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "warn");
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH", "json");
    auto logFilePath = (_tempDir / "hipdnn.log").string();
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", logFilePath.c_str());

    auto descriptor = createAndFinalizeGraph();

    auto jsonFiles = getJsonFilesInDir(_tempDir);
    EXPECT_TRUE(jsonFiles.empty());
}

TEST_F(TestGraphLogger, GraphLoggedWhenEnabled)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH", "json");
    auto logFilePath = (_tempDir / "hipdnn.log").string();
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", logFilePath.c_str());

    auto descriptor = createAndFinalizeGraph();

    auto jsonFiles = getJsonFilesInDir(_tempDir);
    ASSERT_EQ(jsonFiles.size(), 1u);

    // Verify filename format: graph_<16 hex chars>.json
    auto filename = jsonFiles[0].filename().string();
    EXPECT_EQ(filename.rfind("graph_", 0), 0u);
    EXPECT_EQ(filename.rfind(".json"), filename.size() - 5);

    // Verify the file contains valid JSON with expected graph fields
    std::ifstream file(jsonFiles[0]);
    ASSERT_TRUE(file.is_open());
    auto j = nlohmann::json::parse(file);
    EXPECT_TRUE(j.contains("compute_data_type"));
    EXPECT_TRUE(j.contains("nodes"));
    EXPECT_TRUE(j.contains("tensors"));
    EXPECT_TRUE(j.contains("name"));
}

TEST_F(TestGraphLogger, DuplicateGraphNotLoggedTwice)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH", "json");
    auto logFilePath = (_tempDir / "hipdnn.log").string();
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", logFilePath.c_str());

    // Finalize the same graph twice
    auto descriptor1 = createAndFinalizeGraph();
    auto descriptor2 = createAndFinalizeGraph();

    auto jsonFiles = getJsonFilesInDir(_tempDir);
    EXPECT_EQ(jsonFiles.size(), 1u);
}

TEST_F(TestGraphLogger, DifferentGraphsLoggedSeparately)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH", "json");
    auto logFilePath = (_tempDir / "hipdnn.log").string();
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", logFilePath.c_str());

    // Create first graph with default data types
    auto descriptor1 = createAndFinalizeGraph();

    // Create a second graph with different data types
    {
        flatbuffers::FlatBufferBuilder builder;
        std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
            tensorAttributes;
        std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
        auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
            builder,
            "different_graph",
            hipdnn_data_sdk::data_objects::DataType::HALF,
            hipdnn_data_sdk::data_objects::DataType::FLOAT,
            hipdnn_data_sdk::data_objects::DataType::FLOAT,
            &tensorAttributes,
            &nodes);
        builder.Finish(graphOffset);
        auto serialized = builder.Release();

        GraphDescriptor descriptor2;
        descriptor2.deserializeGraph(serialized.data(), serialized.size());
        auto handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
        descriptor2.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
        descriptor2.finalize();
    }

    auto jsonFiles = getJsonFilesInDir(_tempDir);
    EXPECT_EQ(jsonFiles.size(), 2u);
}

TEST_F(TestGraphLogger, OutputDirectoryDerivedFromLogFile)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH", "json");

    // Create a subdirectory and point HIPDNN_LOG_FILE there
    auto subDir = _tempDir / "logs";
    std::filesystem::create_directories(subDir);
    auto logFilePath = (subDir / "hipdnn.log").string();
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", logFilePath.c_str());

    auto descriptor = createAndFinalizeGraph();

    // JSON should be in the same directory as HIPDNN_LOG_FILE
    auto jsonFiles = getJsonFilesInDir(subDir);
    EXPECT_EQ(jsonFiles.size(), 1u);

    // And NOT in the temp dir root
    auto rootJsonFiles = getJsonFilesInDir(_tempDir);
    EXPECT_TRUE(rootJsonFiles.empty());
}

TEST_F(TestGraphLogger, GraphLogModeResetOnShutdown)
{
    // Enable graph logging
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH", "json");

    EXPECT_TRUE(hipdnn_data_sdk::logging::isGraphLoggingEnabled());

    // Shutdown resets the cache
    hipdnn_backend::logging::loggerShutdown();

    // Now change the env var
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_GRAPH");

    // After shutdown + env var change, the mode should be re-read as OFF
    EXPECT_FALSE(hipdnn_data_sdk::logging::isGraphLoggingEnabled());
}
