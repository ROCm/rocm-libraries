// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/GraphTestUtils.hpp"
#include "logging/GraphLogger.hpp"
#include "logging/Logging.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <nlohmann/json.hpp>
#include <thread>

using namespace hipdnn_backend;

class TestGraphLogger : public ::testing::Test
{
protected:
    std::filesystem::path _tempDir;
    std::string _tempDirStr;

    std::unique_ptr<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _logGraphDirGuard;

    void SetUp() override
    {
        _logGraphDirGuard
            = std::make_unique<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter>(
                "HIPDNN_LOG_GRAPH_DIR");

        hipdnn_backend::logging::loggerShutdown();

        // Create a unique temp directory for each test
        _tempDir
            = std::filesystem::temp_directory_path()
              / ("hipdnn_graph_test_"
                 + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + "_"
                 + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(_tempDir);
        _tempDirStr = _tempDir.string();

        testing::internal::CaptureStderr();
    }

    void TearDown() override
    {
        hipdnn_backend::logging::loggerShutdown();

        testing::internal::GetCapturedStderr();

        _logGraphDirGuard.reset();

        if(std::filesystem::exists(_tempDir))
        {
            std::filesystem::remove_all(_tempDir);
        }
    }

    static GraphDescriptor createAndFinalizeGraph(hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
    {
        auto bundle = test_utilities::createDefaultConvOp(computeType);

        GraphDescriptor descriptor;
        const std::array<HipdnnBackendDescriptor*, 1> ops = {bundle.convOp.get()};
        descriptor.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                1,
                                static_cast<const void*>(ops.data()));

        auto handle = reinterpret_cast<hipdnnHandle_t>(0x12345678);
        descriptor.setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                HIPDNN_TYPE_HANDLE,
                                1,
                                static_cast<const void*>(&handle));
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
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_GRAPH_DIR");
    hipdnn_backend::logging::loggerShutdown();

    auto descriptor = createAndFinalizeGraph();

    auto jsonFiles = getJsonFilesInDir(_tempDir);
    EXPECT_TRUE(jsonFiles.empty());
}

TEST_F(TestGraphLogger, GraphLoggedWhenEnabled)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH_DIR", _tempDirStr.c_str());
    hipdnn_backend::logging::loggerShutdown();

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
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH_DIR", _tempDirStr.c_str());
    hipdnn_backend::logging::loggerShutdown();

    // Finalize the same graph twice
    auto descriptor1 = createAndFinalizeGraph();
    auto descriptor2 = createAndFinalizeGraph();

    auto jsonFiles = getJsonFilesInDir(_tempDir);
    EXPECT_EQ(jsonFiles.size(), 1u);
}

TEST_F(TestGraphLogger, DifferentGraphsLoggedSeparately)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH_DIR", _tempDirStr.c_str());
    hipdnn_backend::logging::loggerShutdown();

    // Create first graph with FLOAT compute type
    auto descriptor1 = createAndFinalizeGraph(HIPDNN_DATA_FLOAT);

    // Create a second graph with HALF compute type (produces different serialized content)
    auto descriptor2 = createAndFinalizeGraph(HIPDNN_DATA_HALF);

    auto jsonFiles = getJsonFilesInDir(_tempDir);
    EXPECT_EQ(jsonFiles.size(), 2u);
}

TEST_F(TestGraphLogger, OutputDirectoryUsesEnvVar)
{
    // Create a subdirectory and point HIPDNN_LOG_GRAPH_DIR there
    auto subDir = _tempDir / "graphs";
    const std::string subDirStr = subDir.string();
    std::filesystem::create_directories(subDir);
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH_DIR", subDirStr.c_str());
    hipdnn_backend::logging::loggerShutdown();

    auto descriptor = createAndFinalizeGraph();

    // JSON should be in the specified directory
    auto jsonFiles = getJsonFilesInDir(subDir);
    EXPECT_EQ(jsonFiles.size(), 1u);

    // And NOT in the temp dir root
    auto rootJsonFiles = getJsonFilesInDir(_tempDir);
    EXPECT_TRUE(rootJsonFiles.empty());
}

TEST_F(TestGraphLogger, OutputDirectoryCreatedIfMissing)
{
    auto newDir = _tempDir / "new_subdir" / "graphs";
    const std::string newDirStr = newDir.string();

    ASSERT_FALSE(std::filesystem::exists(newDir));

    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH_DIR", newDirStr.c_str());
    hipdnn_backend::logging::loggerShutdown();

    auto descriptor = createAndFinalizeGraph();

    EXPECT_TRUE(std::filesystem::exists(newDir));
    auto jsonFiles = getJsonFilesInDir(newDir);
    EXPECT_EQ(jsonFiles.size(), 1u);
}

TEST_F(TestGraphLogger, RelativePathResolvedFromCurrentDirectory)
{
    // Use a relative path under the temp dir to avoid polluting the working directory
    auto relativeSubdir
        = std::string("hipdnn_rel_test_")
          + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

    // Change the working directory to _tempDir so the relative path resolves inside it
    auto originalCwd = std::filesystem::current_path();
    std::filesystem::current_path(_tempDir);

    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH_DIR", relativeSubdir.c_str());
    hipdnn_backend::logging::loggerShutdown();

    auto descriptor = createAndFinalizeGraph();

    // Restore the original working directory before assertions
    std::filesystem::current_path(originalCwd);

    auto expectedDir = _tempDir / relativeSubdir;
    EXPECT_TRUE(std::filesystem::exists(expectedDir));
    auto jsonFiles = getJsonFilesInDir(expectedDir);
    EXPECT_EQ(jsonFiles.size(), 1u);
}

TEST_F(TestGraphLogger, CacheResetOnShutdown)
{
    // Enable graph logging
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_GRAPH_DIR", _tempDirStr.c_str());
    hipdnn_backend::logging::loggerShutdown();

    EXPECT_TRUE(logging::GraphLogger::isEnabled());

    // Shutdown resets the cache
    hipdnn_backend::logging::loggerShutdown();

    // Now unset the env var
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_GRAPH_DIR");

    // After shutdown + env var change, isEnabled should return false
    EXPECT_FALSE(logging::GraphLogger::isEnabled());
}
