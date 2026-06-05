// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestDuplicatePolicyIdAPlugin.cpp
 * @brief Test plugin with duplicate policy ID (plugin A)
 *
 * This plugin returns the same policy name as TestDuplicatePolicyIdBPlugin,
 * which generates the same policy ID (via FNV-1a hash of the name).
 * This triggers HeuristicPluginManager::validateBeforeAdding() duplicate ID check
 * (lines 70-78 in HeuristicPluginManager.hpp).
 */

#include "TestHeuristicPluginBase.hpp"

#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class DuplicatePolicyIdAPlugin : public TestHeuristicPluginBase
{
public:
    const char* getPolicyName() const override
    {
        // Same name as plugin B -> same policy ID
        return "TestDuplicatePolicyName";
    }

    const char* getPluginVersion() const override
    {
        return "1.0.0-duplicate-a";
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestHeuristicPluginBase::setInstance(std::make_unique<DuplicatePolicyIdAPlugin>());
}

// Register all API functions using the macro
REGISTER_HEURISTIC_PLUGIN_API()
