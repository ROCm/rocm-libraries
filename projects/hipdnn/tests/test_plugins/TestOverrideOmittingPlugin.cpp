// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Fake plugin that implements only the standard execute entry.

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

// Define thread-local LastCallRecord storage and the suffixed C-API
// observation entry points for tests. Must come before the plugin class so
// the override of `lastCallRecord()` can name the suffixed accessor.
DEFINE_TEST_PLUGIN_LAST_CALL_STORAGE(OverrideOmitting)

class OverrideOmittingPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_OverrideOmittingPlugin";
    }
    const char* getPluginVersion() const override
    {
        return "1.0.0";
    }

    /// Reports the baseline plugin SDK version.
    const char* getPluginApiVersion() const override
    {
        return apiVersionWithoutTweak();
    }

    int64_t getEngineId() const override
    {
        return hipdnn_tests::plugin_constants::engineId<OverrideOmittingPlugin>();
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
    /// `getLastCallRecord_OverrideOmitting()`.
    TestPluginLastCallRecord& lastCallRecord() const override
    {
        return testPluginLastCallRecord_OverrideOmitting();
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<OverrideOmittingPlugin>());
}

// Register only the standard plugin API functions.
REGISTER_TEST_PLUGIN_API()
