// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

// Reports its engine applicable for every graph, then cannot build a plan for
// it and says so with HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE from plan creation.
// Mirrors a provider whose applicability check is a cheap op-level predicate
// while the real solver search only happens when the execution plan is created.
class NoSolverPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_NoSolverPlugin";
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
        return hipdnn_tests::plugin_constants::engineId<NoSolverPlugin>();
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
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
                                                       "NoSolverPlugin: no applicable solver");
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<NoSolverPlugin>());
}

REGISTER_TEST_PLUGIN_API()
