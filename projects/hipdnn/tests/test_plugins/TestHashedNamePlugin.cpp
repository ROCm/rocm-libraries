// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

// A well-behaved plugin whose engine id is the hash of its own engine name, the
// identity HIPDNN_REGISTER_ENGINE gives production plugins.
class HashedNamePlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_HashedNamePlugin";
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
        return hipdnn_tests::plugin_constants::engineId<HashedNamePlugin>();
    }
    const char* getEngineName() const override
    {
        return hipdnn_tests::plugin_constants::K_HASHED_NAME_PLUGIN_ENGINE_NAME;
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
    TestPluginBase::setInstance(std::make_unique<HashedNamePlugin>());
}

REGISTER_TEST_PLUGIN_API()
REGISTER_TEST_PLUGIN_ENGINE_NAME_API()
