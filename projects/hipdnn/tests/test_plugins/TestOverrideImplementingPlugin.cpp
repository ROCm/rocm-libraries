// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Fake plugin that implements both standard and override-aware execute entries.

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

// Define thread-local LastCallRecord storage and the suffixed C-API
// observation entry points for tests. Must come before the plugin class so
// the override of `lastCallRecord()` can name the suffixed accessor.
DEFINE_TEST_PLUGIN_LAST_CALL_STORAGE(OverrideImplementing)

class OverrideImplementingPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_OverrideImplementingPlugin";
    }
    const char* getPluginVersion() const override
    {
        return "1.0.0";
    }

    /// Reports the minimum plugin SDK version for override execution.
    const char* getPluginApiVersion() const override
    {
        return hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION.data();
    }

    int64_t getEngineId() const override
    {
        return hipdnn_tests::plugin_constants::engineId<OverrideImplementingPlugin>();
    }
    uint32_t getNumEngines() const override
    {
        return 1;
    }
    uint32_t getNumApplicableEngines() const override
    {
        return 1;
    }

    /// Routes the base-class observation hooks to this plugin's suffixed
    /// thread-local storage so tests can inspect dispatch via
    /// `getLastCallRecord_OverrideImplementing()`.
    TestPluginLastCallRecord& lastCallRecord() const override
    {
        return testPluginLastCallRecord_OverrideImplementing();
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<OverrideImplementingPlugin>());
}

// Register all standard plugin API functions PLUS the optional override
// execute entry. Presence of the override symbol is exactly what makes
// this plugin "override-implementing" from the host's perspective.
REGISTER_TEST_PLUGIN_API()
REGISTER_TEST_PLUGIN_OVERRIDE_API()
