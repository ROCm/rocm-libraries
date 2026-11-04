// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

// A plugin that names its engine well but hardcodes an engine id that the name
// does not hash back to. The host logs an error and drops the engine at load.
//
// It exists as its own fixture so the tests asserting that error load nothing
// else, and so no other test has to tolerate the error in its logs.
class MismatchedNamePlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_MismatchedNamePlugin";
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
        return hipdnn_tests::plugin_constants::engineId<MismatchedNamePlugin>();
    }
    const char* getEngineName() const override
    {
        return hipdnn_tests::plugin_constants::K_MISMATCHED_NAME_PLUGIN_ENGINE_NAME;
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
    TestPluginBase::setInstance(std::make_unique<MismatchedNamePlugin>());
}

REGISTER_TEST_PLUGIN_API()
REGISTER_TEST_PLUGIN_ENGINE_NAME_API()
