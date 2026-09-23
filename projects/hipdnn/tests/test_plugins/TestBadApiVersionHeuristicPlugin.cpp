// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestBadApiVersionHeuristicPlugin.cpp
 * @brief Test plugin that returns incompatible API version
 *
 * This plugin intentionally returns an API version with wrong major version
 * to trigger HeuristicPluginManager::validateBeforeAdding() API version check
 * (lines 55-66 in HeuristicPluginManager.hpp).
 */

#include "TestHeuristicPluginBase.hpp"

#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class BadApiVersionHeuristicPlugin : public TestHeuristicPluginBase
{
public:
    const char* getPolicyName() const override
    {
        return "TestBadApiVersionPolicy";
    }

    const char* getPluginVersion() const override
    {
        return "1.0.0";
    }

    const char* getApiVersion() const override
    {
        // Return incompatible API version (wrong major version)
        return "99.0.0";
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestHeuristicPluginBase::setInstance(std::make_unique<BadApiVersionHeuristicPlugin>());
}

// Register all API functions using the macro
REGISTER_HEURISTIC_PLUGIN_API()
