// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

// A plugin that exports the optional engine-name entry point and then answers
// it badly, one malformed answer per engine id:
//
//   LyingEngineNamePlugin            SUCCESS with *name left null
//   LyingEngineNamePluginEmptyName   SUCCESS with *name set to ""
//   LyingEngineNamePluginErrorStatus a failure status with *name set anyway
//
// All three ids are published, so admission rules on each: the first two answers
// decline and those engines load, while the failure status is a defect and costs
// that engine its place. Only the first id backs an applicable engine -- the entry
// point is keyed by engine id, so it needs no engine details.
class LyingEngineNamePlugin : public TestPluginBase
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
        return hipdnn_tests::plugin_constants::engineId<LyingEngineNamePlugin>();
    }

    // getEngineName is deliberately not overridden, so EngineDetails.name is unset
    // and resolution has to reach the host-side fallback.

    uint32_t getNumEngines() const override
    {
        return 3;
    }

    std::vector<int64_t> getAllEngineIds() const override
    {
        namespace constants = hipdnn_tests::plugin_constants;

        return {constants::engineId<LyingEngineNamePlugin>(),
                constants::engineId<LyingEngineNamePluginEmptyName>(),
                constants::engineId<LyingEngineNamePluginErrorStatus>()};
    }
    uint32_t getNumApplicableEngines() const override
    {
        return 1;
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<LyingEngineNamePlugin>());
}

// The standard surface comes from the shared macro. The engine-name entry point
// is hand-written rather than taken from REGISTER_TEST_PLUGIN_ENGINE_NAME_API()
// because that macro delegates to the conforming implementation.
REGISTER_TEST_PLUGIN_API()

// NOLINTBEGIN(readability-identifier-naming)
extern "C" {

HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetEngineName(int64_t engineId, const char** name)
{
    namespace constants = hipdnn_tests::plugin_constants;

    if(name == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    if(engineId == constants::engineId<LyingEngineNamePlugin>())
    {
        // Claims success without ever writing a name. Set explicitly so the
        // fixture does not depend on the caller having zeroed it first.
        *name = nullptr;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    if(engineId == constants::engineId<LyingEngineNamePluginEmptyName>())
    {
        *name = "";
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    if(engineId == constants::engineId<LyingEngineNamePluginErrorStatus>())
    {
        // The host must honour the status and discard the name.
        *name = constants::K_LYING_ENGINE_NAME_UNUSABLE_NAME;
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }

    return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
}

} // extern "C"
// NOLINTEND(readability-identifier-naming)
