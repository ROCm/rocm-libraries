// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "common/PlatformUtils.hpp"
#include "harness/DeviceArch.hpp"
#include "harness/TestSettings.hpp"

namespace hipdnn_integration_tests
{

// Methods for determining acceptable tolerance when comparing reference
// implementation output to the selected engine's output.
enum class ToleranceMode
{
    DEFAULT,
};

// Selects the reference executor used for graph validation.
enum class ReferenceExecutorType
{
    CPU,
    GPU,
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
                           std::optional<std::filesystem::path> configPath = std::nullopt,
                           std::optional<ReferenceExecutorType> referenceExecutorType
                           = std::nullopt)
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
        instance._referenceExecutorType = referenceExecutorType;

        // If CLI didn't provide a value, check env var once at init
        if(!instance._referenceExecutorType.has_value())
        {
            auto val = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_REFERENCE_EXECUTOR");
            if(!val.empty())
            {
                std::transform(val.begin(), val.end(), val.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
                if(val == "gpu")
                {
                    instance._referenceExecutorType = ReferenceExecutorType::GPU;
                }
                else if(val == "cpu")
                {
                    instance._referenceExecutorType = ReferenceExecutorType::CPU;
                }
                else
                {
                    throw std::runtime_error("Invalid HIPDNN_TEST_REFERENCE_EXECUTOR value '" + val
                                             + "'; expected 'cpu' or 'gpu'");
                }
            }
        }

        if(configPath.has_value())
        {
            instance._testSettings.emplace(*configPath);
        }

        // Detect device 0's gfx arch once at startup. Used by [[test_skips]]
        // todo: In future allow the test runner to use any specified device.
        instance._currentArch = currentDeviceArchRaw();

        // Detect platform once at startup (always succeeds; PlatformUtils.hpp
        // refuses to compile on unsupported OSes).
        instance._currentPlatform = currentPlatform();

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

    // Raw gcnArchName for device 0 detected at init time (e.g.
    // "gfx942:sramecc+:xnack-"). Empty if detection failed.
    const std::string& getCurrentArch() const
    {
        throwIfNotInitialized();
        return _currentArch;
    }

    // Lowercase platform name detected at init time ("windows" or "linux").
    const std::string& getCurrentPlatform() const
    {
        throwIfNotInitialized();
        return _currentPlatform;
    }

    // Find a [[test_skips]] entry matching the given test name on the current
    // device arch and platform. Returns the entry's reason, or std::nullopt
    // if no config is loaded or no entry matches.
    std::optional<std::string> findSkipForTest(std::string_view testName) const
    {
        throwIfNotInitialized();
        if(!_testSettings.has_value())
        {
            return std::nullopt;
        }
        return _testSettings->findSkip(testName, _currentArch, _currentPlatform);
    }

    // Get the reference executor type. Value is resolved once at init time:
    // CLI flag > HIPDNN_TEST_REFERENCE_EXECUTOR env var > CPU default.
    ReferenceExecutorType getReferenceExecutorType() const
    {
        throwIfNotInitialized();
        return _referenceExecutorType.value_or(ReferenceExecutorType::CPU);
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
    std::optional<ReferenceExecutorType> _referenceExecutorType;
    std::string _currentArch;
    std::string _currentPlatform;
    bool _failOnUnsupported = false;
    bool _skipGraphValidation = false;
    bool _initialized = false;
};

} // namespace hipdnn_integration_tests
