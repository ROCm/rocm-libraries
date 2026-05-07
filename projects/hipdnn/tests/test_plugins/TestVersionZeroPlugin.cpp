// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Fake plugin for RFC 0008 plan T-missing #2: reports a parseable
// but too-low API version string ("0.0.0"). The host's load-time
// `parsedApiVersion()` cache engages successfully (distinguishing this
// case from the malformed-version plugin where the parse THROWS), but
// the version comparison still rejects the plugin against the post-PR1
// baseline of "1.0.0". This pins the "parsed-but-too-low" rejection path
// as a separate code path from the malformed/unparseable rejection.

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class VersionZeroPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_VersionZeroPlugin";
    }
    const char* getPluginVersion() const override
    {
        return "1.0.0";
    }

    /// Reports a parseable but too-low API version. Distinct from the
    /// malformed-version plugin (which throws during parse). The host's
    /// load-time version filter must reject this plugin before it reaches
    /// any dispatch path.
    const char* getPluginApiVersion() const override
    {
        return "0.0.0";
    }

    int64_t getEngineId() const override
    {
        return hipdnn_tests::plugin_constants::engineId<VersionZeroPlugin>();
    }
    uint32_t getNumEngines() const override
    {
        return 1;
    }
    uint32_t getNumApplicableEngines() const override
    {
        return 1;
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<VersionZeroPlugin>());
}

// Register ONLY the standard plugin API functions. The override-execute
// API is intentionally NOT registered; the plugin is expected to be
// filtered out by the version baseline before either entry could be
// invoked.
REGISTER_TEST_PLUGIN_API()
