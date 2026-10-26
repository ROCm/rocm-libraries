// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

// Reports its engine applicable for every graph, then fails to create the execution plan
// with a resource failure (HIPDNN_PLUGIN_STATUS_ALLOC_FAILED). Unlike the no-solver fake,
// this is a fault rather than a decline, so callers must stop instead of trying another
// engine.
class ResourceFailPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_ResourceFailPlugin";
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
        return hipdnn_tests::plugin_constants::engineId<ResourceFailPlugin>();
    }
    uint32_t getNumEngines() const override
    {
        return 1;
    }
    uint32_t getNumApplicableEngines() const override
    {
        return 1;
    }

    void onCreateExecutionContext() const override
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_ALLOC_FAILED,
                                                       "ResourceFailPlugin: out of memory");
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<ResourceFailPlugin>());
}

REGISTER_TEST_PLUGIN_API()
