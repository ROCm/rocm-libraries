// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"
// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class ExecuteFailsPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_ExecuteFailsPlugin";
    }
    const char* getPluginVersion() const override
    {
        return "1.0.0";
    }
    int64_t getEngineId() const override
    {
        return hipdnn_tests::plugin_constants::engineId<ExecuteFailsPlugin>();
    }
    uint32_t getNumEngines() const override
    {
        return 1;
    }
    uint32_t getNumApplicableEngines() const override
    {
        return 1;
    }

    // Override executeGraph to simulate execution failure
    void executeGraph() const override
    {
        throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                   "Simulated execution failure for testing");
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<ExecuteFailsPlugin>());
}

// Register all API functions
REGISTER_TEST_PLUGIN_API()
