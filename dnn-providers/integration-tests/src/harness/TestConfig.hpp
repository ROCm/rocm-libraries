// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "harness/TestSettings.hpp"

namespace hipdnn_integration_tests
{

// Methods for determining acceptable tolerance when comparing reference
// implementation output to the selected engine's output.
enum class ToleranceMode
{
    DEFAULT,
};

// Singleton class for storing CLI-based test configuration.
// All arguments are independently optional:
//   - articlePath: omit to use hipDNN's default plugin discovery
//   - engineName: omit to let hipDNN select the engine
//   - failOnUnsupported: when true, FAIL instead of SKIP for unsupported graphs
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
    static void initialize(std::optional<std::filesystem::path> articlePath,
                           std::optional<std::string> engineName,
                           bool failOnUnsupported = false,
                           bool skipGraphValidation = false,
                           std::optional<std::filesystem::path> configPath = std::nullopt)
    {
        TestConfig& instance = get();
        if(instance._initialized)
        {
            throw std::runtime_error("TestConfig::initialize() called more than once");
        }
        instance._articlePath = std::move(articlePath);
        instance._engineName = std::move(engineName);
        instance._failOnUnsupported = failOnUnsupported;
        instance._skipGraphValidation = skipGraphValidation;

        if(configPath.has_value())
        {
            instance._testSettings.emplace(*configPath);
        }

        instance._initialized = true;
    }

    bool hasArticlePath() const
    {
        throwIfNotInitialized();
        return _articlePath.has_value();
    }

    bool hasEngineName() const
    {
        throwIfNotInitialized();
        return _engineName.has_value();
    }

    bool failOnUnsupported() const
    {
        throwIfNotInitialized();
        return _failOnUnsupported;
    }

    bool skipGraphValidation() const
    {
        throwIfNotInitialized();
        return _skipGraphValidation;
    }

    // Get the article (plugin .so) path. Throws if not provided.
    const std::filesystem::path& getArticlePath() const
    {
        throwIfNotInitialized();
        if(!_articlePath.has_value())
        {
            throw std::runtime_error("getArticlePath() called but --test-article was not provided");
        }
        return _articlePath.value();
    }

    // Get the engine name string. Throws if not provided.
    std::string_view getEngineName() const
    {
        throwIfNotInitialized();
        if(!_engineName.has_value())
        {
            throw std::runtime_error("getEngineName() called but --test-engine was not provided");
        }
        return _engineName.value();
    }

    // Get the engine ID from the engine name. Throws if engine not provided.
    int64_t getEngineId() const
    {
        throwIfNotInitialized();
        if(!_engineName.has_value())
        {
            throw std::runtime_error("getEngineId() called but --test-engine was not provided");
        }
        return hipdnn_data_sdk::utilities::engineNameToId(_engineName.value());
    }

    // Get tolerance mode (always DEFAULT since only one mode exists)
    ToleranceMode getToleranceMode() const
    {
        throwIfNotInitialized();
        return ToleranceMode::DEFAULT;
    }

    // Check if a test settings file was provided
    bool hasTestSettings() const
    {
        throwIfNotInitialized();
        return _testSettings.has_value();
    }

    // Find a tolerance override matching the given test name.
    // Returns std::nullopt if no config loaded or no filter matches.
    std::optional<ToleranceOverride> findToleranceOverride(std::string_view testName) const
    {
        throwIfNotInitialized();
        if(!_testSettings.has_value())
        {
            return std::nullopt;
        }
        return _testSettings->findToleranceOverride(testName);
    }

private:
    TestConfig() = default;

    void throwIfNotInitialized() const
    {
        if(!_initialized)
        {
            throw std::runtime_error("TestConfig not initialized");
        }
    }

    std::optional<std::filesystem::path> _articlePath;
    std::optional<std::string> _engineName;
    std::optional<TestSettings> _testSettings;
    bool _failOnUnsupported = false;
    bool _skipGraphValidation = false;
    bool _initialized = false;
};

} // namespace hipdnn_integration_tests
