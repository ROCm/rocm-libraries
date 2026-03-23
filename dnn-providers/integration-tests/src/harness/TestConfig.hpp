// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>

namespace hipdnn_integration_tests {

// Methods for determining acceptable tolerance when comparing reference
// implementation output to the selected engine's output.
//
// Example:
//   [plugins.miopen]
//   name = "miopen_provider_plugin"
//   engines = ["MIOPEN_PLUGIN"]
//
//   [engines.MIOPEN_PLUGIN]
//   tolerance = "gh_12678_tolerance_workaround"
//
// The example config would map to ToleranceMode::GH_12678_TOLERANCE_WORKAROUND
// which uses default tolerance for all graphs besides batch norm backwards
// operating on bfloat16 where it returns a wider tolerance.
enum class ToleranceMode {
    Default,
};

// Singleton class for reading test configuration from JSON file.
class TestConfig {
   public:
    // Get singleton instance
    static TestConfig& get() {
        static TestConfig instance;
        return instance;
    }

    TestConfig(const TestConfig&) = delete;
    TestConfig& operator=(const TestConfig&) = delete;
    TestConfig(TestConfig&&) = delete;
    TestConfig& operator=(TestConfig&&) = delete;

    // Get tolerance mode for a given engine ID.
    ToleranceMode getToleranceMode(int64_t engineId) const {
        std::string engineName;
        try {
            engineName = std::string(hipdnn_data_sdk::utilities::getEngineNameFromId(engineId));
        } catch (const std::out_of_range&) {
            engineName = "Engine" + std::to_string(engineId);
        }

        if (auto it = _engineTolerances.find(engineName); it != _engineTolerances.end()) {
            return it->second;
        }
        return ToleranceMode::Default;
    }

    // Check if a full GTest test name is in the expected failures list.
    // Test name format: "TestSuite/Prefix.TestName/ParamName"
    bool isExpectedFailure(const std::string& testName) {
        return _expectedFailures.count(testName) > 0;
    }

    // Get expected plugin names from config (e.g., {"fusilli_plugin",
    // "miopen_provider_plugin"})
    const std::set<std::string>& getExpectedPluginNames() const {
        return _expectedPluginNames;
    }

   private:
    TestConfig() {
        // Get config path
        const char* configPathEnv = std::getenv("HIPDNN_TEST_CONFIG_PATH");
        if (configPathEnv == nullptr || std::strlen(configPathEnv) == 0) {
            throw std::runtime_error("HIPDNN_TEST_CONFIG_PATH environment variable not set");
        }

        // Parse config
        std::filesystem::path configPath = std::filesystem::weakly_canonical(configPathEnv);
        std::ifstream configFile(configPath);
        if (!configFile.is_open()) {
            throw std::runtime_error("Failed to open config file: " + configPath.string());
        }
        try {
            auto config = nlohmann::json::parse(configFile);

            // Populate expected failures set from flattened list
            if (config.contains("expected_failures")) {
                for (const auto& name : config["expected_failures"]) {
                    _expectedFailures.insert(name.get<std::string>());
                }
            }

            // Populate expected plugin names from plugin definitions
            if (config.contains("plugins")) {
                for (const auto& [name, info] : config["plugins"].items()) {
                    if (info.contains("name")) {
                        _expectedPluginNames.insert(info["name"].get<std::string>());
                    }
                }
            }

            // Populate engine tolerance modes
            if (config.contains("engines")) {
                for (const auto& [engineName, engineConfig] : config["engines"].items()) {
                    if (engineConfig.contains("tolerance")) {
                        auto val = engineConfig["tolerance"].get<std::string>();
                        if (val == "default") {
                            _engineTolerances[engineName] = ToleranceMode::Default;
                        } else {
                            throw std::runtime_error("Unknown tolerance mode: " + val);
                        }
                    }
                }
            }
        } catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error("Failed to parse config JSON: " + std::string(e.what()));
        }
    }

    std::set<std::string> _expectedFailures;
    std::set<std::string> _expectedPluginNames;
    std::map<std::string, ToleranceMode> _engineTolerances;
};

}  // namespace hipdnn_integration_tests
