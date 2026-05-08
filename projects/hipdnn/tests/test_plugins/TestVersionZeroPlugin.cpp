// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Fake plugin that reports a parseable but too-low API version string.

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

    /// Reports a parseable but too-low API version.
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

// Register only the standard plugin API functions.
REGISTER_TEST_PLUGIN_API()
