// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Fake plugin that reports the runtime pass-by-value minimum API version
// (K_PASS_BY_VALUE_MIN_API_VERSION, "1.2.0"). It generically serves any graph
// like TestGoodPlugin, so it is the positive counterpart to the pre-1.2.0
// fakes in the runtime pass-by-value version-filter tests: a runtime
// pass-by-value graph is admitted for this plugin and rejected for the others.

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class PassByValuePlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_PassByValuePlugin";
    }
    const char* getPluginVersion() const override
    {
        return "1.0.0";
    }

    /// Reports the minimum plugin SDK version that advertises runtime pass-by-value support.
    const char* getPluginApiVersion() const override
    {
        return hipdnn_plugin_sdk::K_PASS_BY_VALUE_MIN_API_VERSION.data();
    }

    int64_t getEngineId() const override
    {
        return hipdnn_tests::plugin_constants::engineId<PassByValuePlugin>();
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
    TestPluginBase::setInstance(std::make_unique<PassByValuePlugin>());
}

// Register all standard plugin API functions
REGISTER_TEST_PLUGIN_API()
