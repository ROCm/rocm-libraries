// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestEmptyNameHeuristicPlugin.cpp
 * @brief Test plugin that returns empty policy name
 *
 * This plugin intentionally returns an empty string for policy name
 * to trigger HeuristicPluginManager::validateBeforeAdding() policy name check
 * (lines 82-89 in HeuristicPluginManager.hpp).
 */

#include "TestHeuristicPluginBase.hpp"

#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class EmptyNameHeuristicPlugin : public TestHeuristicPluginBase
{
public:
    const char* getPolicyName() const override
    {
        // Return empty policy name to trigger validation error
        return "";
    }

    const char* getPluginVersion() const override
    {
        return "1.0.0";
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestHeuristicPluginBase::setInstance(std::make_unique<EmptyNameHeuristicPlugin>());
}

// Register all API functions using the macro
REGISTER_HEURISTIC_PLUGIN_API()
