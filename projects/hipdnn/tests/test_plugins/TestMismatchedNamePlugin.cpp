// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

// A plugin that names its engine well but hardcodes an engine id that the name
// does not hash back to, which is the defect the host reports through its
// name/id disagreement warning.
//
// The host raises that warning only once per (plugin, engine id) per process,
// so the fixture asserting it must be loaded by exactly one test. Other fakes
// carry the same disagreement but are loaded by several tests, which would
// leave the assertion dependent on test order.
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

// The optional engine-name entry point is registered alongside the standard
// surface: tier 1 is where the warning is raised from.
REGISTER_TEST_PLUGIN_API()
REGISTER_TEST_PLUGIN_ENGINE_NAME_API()
