// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

// A well-behaved plugin whose engine id is the hash of its own engine name.
// Production plugins get that identity from HIPDNN_REGISTER_ENGINE, which
// derives the id from the name; the other fake plugins hardcode unrelated ids.
// Without a fake that honours the identity, no test can exercise a filter that
// resolves an engine name by hashing it, such as deselect_engines(names).
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

// Register all standard plugin API functions PLUS the optional engine-name
// entry point, so the engine name reaches the backend through tier 1 of the
// name-resolution chain as well as through EngineDetails.name.
REGISTER_TEST_PLUGIN_API()
REGISTER_TEST_PLUGIN_ENGINE_NAME_API()
