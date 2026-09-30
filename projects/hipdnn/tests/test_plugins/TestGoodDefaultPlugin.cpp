// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"
// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class GoodDefaultPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return HIPDNN_COMPONENT_NAME;
    }
    const char* getPluginVersion() const override
    {
        return "1.0.0";
    }

    const char* getPluginApiVersion() const override
    {
        return apiVersionWithoutTweak();
    }

    int64_t getEngineId() const override
    {
        return hipdnn_tests::plugin_constants::engineId<GoodDefaultPlugin>();
    }
    const char* getEngineName() const override
    {
        return hipdnn_tests::plugin_constants::K_GOOD_DEFAULT_PLUGIN_ENGINE_NAME;
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
    TestPluginBase::setInstance(std::make_unique<GoodDefaultPlugin>());
}

// The only plugin installed to test_plugins/default, so it is the one discovered by
// directory scans that need a named engine.
REGISTER_TEST_PLUGIN_API()
REGISTER_TEST_PLUGIN_ENGINE_NAME_API()
