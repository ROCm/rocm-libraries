// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Fake plugin for RFC 0008 Phase 1, Test #3 / #20: reports an API version
// >= K_PHASE1_OVERRIDE_MIN_VERSION (so the applicability version filter does
// NOT skip it) but DOES NOT export the override-execute symbol. The host
// must then catch the discrepancy at dispatch time via the
// `EnginePlugin::hasOverrideExecute()` safety net (RFC §4.6, §7.2) and
// return `HIPDNN_STATUS_NOT_SUPPORTED` rather than crash on a missing
// symbol.

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
DEFINE_TEST_PLUGIN_LAST_CALL_STORAGE(VersionLiar)

class VersionLiarPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_VersionLiarPlugin";
    }
    const char* getPluginVersion() const override
    {
        return "1.0.0";
    }

    /// Reports the centralized RFC 0008 §4.5 placeholder min version even
    /// though this plugin does NOT export the override-execute symbol.
    /// This intentional mismatch exercises the dispatch-time safety net.
    const char* getPluginApiVersion() const override
    {
        return hipdnn_plugin_sdk::K_PHASE1_OVERRIDE_MIN_VERSION.data();
    }

    int64_t getEngineId() const override
    {
        return hipdnn_tests::plugin_constants::engineId<VersionLiarPlugin>();
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
    /// `getLastCallRecord_VersionLiar()`.
    TestPluginLastCallRecord& lastCallRecord() const override
    {
        return testPluginLastCallRecord_VersionLiar();
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<VersionLiarPlugin>());
}

// Register ONLY the standard plugin API functions. Deliberately do NOT
// invoke REGISTER_TEST_PLUGIN_OVERRIDE_API() — that omission is exactly
// the "version lie" this plugin embodies and is verified at build time by
// Test #20 via `nm`.
REGISTER_TEST_PLUGIN_API()
