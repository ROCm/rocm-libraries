// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Engines for the cuDNN v8 shim integration tests. E1 exposes one knob the shim
// projects onto a cuDNN knob type (test.tile_size -> TILE_SIZE) next to two it must
// hide (one with explicit valid values, one with no cuDNN counterpart), serializes
// plans, and records the engine configs it receives. E2 fails at execution-context
// creation, so a plan for it fails to build while its engine config still finalizes.

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

#include <hipdnn_plugin_sdk/KnobFactory.hpp>

#include <cstdint>
#include <string>
#include <vector>

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

namespace
{

constexpr int64_t K_KNOB_ENGINE_ID = hipdnn_tests::plugin_constants::engineId<CudnnShimV8Plugin>();
constexpr int64_t K_CONTEXT_FAILS_ENGINE_ID
    = hipdnn_tests::plugin_constants::engineId<CudnnShimV8PluginContextFails>();

} // namespace

DEFINE_TEST_PLUGIN_KNOB_RECORDER()

class CudnnShimV8Plugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_CudnnShimV8Plugin";
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
        return K_KNOB_ENGINE_ID;
    }

    uint32_t getNumEngines() const override
    {
        return 2;
    }

    std::vector<int64_t> getAllEngineIds() const override
    {
        return {K_KNOB_ENGINE_ID, K_CONTEXT_FAILS_ENGINE_ID};
    }

    uint32_t getNumApplicableEngines() const override
    {
        return 2;
    }

    int64_t describedEngineId(int64_t requestedId) const override
    {
        if(requestedId != K_KNOB_ENGINE_ID && requestedId != K_CONTEXT_FAILS_ENGINE_ID)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                           "Unknown engine id "
                                                               + std::to_string(requestedId));
        }
        return requestedId;
    }

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>>
        engineKnobs(flatbuffers::FlatBufferBuilder& builder, int64_t engineId) const override
    {
        if(engineId != K_KNOB_ENGINE_ID)
        {
            return {};
        }
        using hipdnn_plugin_sdk::KnobFactory;
        return {
            // Plain min/max/step: the shim shows it as TILE_SIZE.
            KnobFactory::createIntKnob(
                builder, "test.tile_size", "Projectable tile size", 16, 0, 64, 8, {}),
            // Maps to BLOCK_SIZE by name, but explicit valid values have no cuDNN
            // representation, so the shim hides it.
            KnobFactory::createIntKnob(
                builder, "test.block_size", "Hidden block size", 8, 0, 0, 1, {8, 16}),
            // No cuDNN knob type matches this name.
            KnobFactory::createIntKnob(
                builder, "test.int_knob", "Unmappable integer knob", 50, 0, 100, 10, {}),
        };
    }

    void checkExecutionContextEngine(int64_t engineId) const override
    {
        if(engineId == K_CONTEXT_FAILS_ENGINE_ID)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "Simulated execution context creation failure for testing");
        }
    }

    void onWorkspaceSizeQuery(const hipdnnPluginConstData_t& engineConfig) const override
    {
        recordTestPluginEngineConfig(&engineConfig);
    }
};

__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<CudnnShimV8Plugin>());
}

REGISTER_TEST_PLUGIN_API()
