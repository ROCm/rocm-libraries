// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestGoodPluginApiVersion.hpp"
#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class GoodPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_GoodPlugin";
    }
    const char* getPluginVersion() const override
    {
        return "1.0.0";
    }

    /// Reports K_TEST_GOOD_PLUGIN_API_VERSION, which supports override-execute but sits
    /// below K_PASS_BY_VALUE_MIN_API_VERSION, so this fake stays the pre-pass-by-value
    /// plugin regardless of the plugin SDK package version.
    const char* getPluginApiVersion() const override
    {
        return hipdnn_tests::plugin_constants::K_TEST_GOOD_PLUGIN_API_VERSION.data();
    }

    int64_t getEngineId() const override
    {
        return hipdnn_tests::plugin_constants::engineId<GoodPlugin>();
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
    TestPluginBase::setInstance(std::make_unique<GoodPlugin>());
}

// The optional engine-name entry point is omitted and `getEngineName()` is left at
// its `nullptr` default, making this the fixture for host-side name fallback.
REGISTER_TEST_PLUGIN_API()
