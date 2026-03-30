// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace hipdnn_integration_tests
{

// Methods for determining acceptable tolerance when comparing reference
// implementation output to the selected engine's output.
enum class ToleranceMode
{
    DEFAULT,
};

// Singleton class for storing CLI-based test configuration.
class TestConfig
{
public:
    // Get singleton instance
    static TestConfig& get()
    {
        static TestConfig s_instance;
        return s_instance;
    }

    TestConfig(const TestConfig&) = delete;
    TestConfig& operator=(const TestConfig&) = delete;
    TestConfig(TestConfig&&) = delete;
    TestConfig& operator=(TestConfig&&) = delete;

    // Initialize with CLI arguments. Must be called before any get() access.
    // Throws if called more than once or if the singleton was already accessed uninitialized.
    static void initialize(std::filesystem::path articlePath, std::string engineName)
    {
        TestConfig& instance = get();
        if(instance._initialized)
        {
            throw std::runtime_error("TestConfig::initialize() called more than once");
        }
        instance._articlePath = std::move(articlePath);
        instance._engineName = std::move(engineName);
        instance._initialized = true;
    }

    // Get the article (plugin .so) path
    const std::filesystem::path& getArticlePath() const
    {
        if(!_initialized)
        {
            throw std::runtime_error("TestConfig not initialized");
        }
        return _articlePath;
    }

    // Get the engine name string
    std::string_view getEngineName() const
    {
        if(!_initialized)
        {
            throw std::runtime_error("TestConfig not initialized");
        }
        return _engineName;
    }

    // Get the engine ID from the engine name
    int64_t getEngineId() const
    {
        if(!_initialized)
        {
            throw std::runtime_error("TestConfig not initialized");
        }
        return hipdnn_data_sdk::utilities::engineNameToId(_engineName);
    }

    // Get tolerance mode (always DEFAULT since only one mode exists)
    ToleranceMode getToleranceMode() const
    {
        if(!_initialized)
        {
            throw std::runtime_error("TestConfig not initialized");
        }

        return ToleranceMode::DEFAULT;
    }

private:
    TestConfig() = default;

    std::filesystem::path _articlePath;
    std::string _engineName;
    bool _initialized = false;
};

} // namespace hipdnn_integration_tests
