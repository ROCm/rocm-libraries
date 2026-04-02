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

    // Initialize with CLI arguments for engine-specific mode.
    // Must be called before any get() access.
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

    // Initialize for Out Of The Box mode (no specific engine or plugin path).
    // hipDNN uses default plugin discovery and selects the engine itself.
    static void initializeOOTB()
    {
        TestConfig& instance = get();
        if(instance._initialized)
        {
            throw std::runtime_error("TestConfig::initialize() called more than once");
        }
        instance._ootbMode = true;
        instance._initialized = true;
    }

    // Returns true if running in OOTB mode (no specific engine selected).
    bool isOOTBMode() const
    {
        if(!_initialized)
        {
            throw std::runtime_error("TestConfig not initialized");
        }
        return _ootbMode;
    }

    // Get the article (plugin .so) path. Throws in OOTB mode.
    const std::filesystem::path& getArticlePath() const
    {
        if(!_initialized)
        {
            throw std::runtime_error("TestConfig not initialized");
        }
        if(_ootbMode)
        {
            throw std::runtime_error("getArticlePath() not available in OOTB mode");
        }
        return _articlePath;
    }

    // Get the engine name string. Throws in OOTB mode.
    std::string_view getEngineName() const
    {
        if(!_initialized)
        {
            throw std::runtime_error("TestConfig not initialized");
        }
        if(_ootbMode)
        {
            throw std::runtime_error("getEngineName() not available in OOTB mode");
        }
        return _engineName;
    }

    // Get the engine ID from the engine name. Throws in OOTB mode.
    int64_t getEngineId() const
    {
        if(!_initialized)
        {
            throw std::runtime_error("TestConfig not initialized");
        }
        if(_ootbMode)
        {
            throw std::runtime_error("getEngineId() not available in OOTB mode");
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
    bool _ootbMode = false;
};

} // namespace hipdnn_integration_tests
