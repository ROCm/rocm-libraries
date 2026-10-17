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
#include <vector>

// Raw flatbuffer bytes of each EngineConfig received at workspace-size queries; read by
// tests through the exported accessors below (see TestPluginKnobRecorder.hpp).
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static thread_local std::vector<std::vector<uint8_t>> gReceivedKnobs;

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

    static hipdnnPluginStatus_t getApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                                       const hipdnnPluginConstData_t* opGraph,
                                                       int64_t* engineIds,
                                                       uint32_t maxEngines,
                                                       uint32_t* numEngines)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", opGraph=" << static_cast<const void*>(opGraph)
                                << ", engineIds=" << static_cast<void*>(engineIds)
                                << ", maxEngines=" << maxEngines
                                << ", numEngines=" << static_cast<void*>(numEngines));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(opGraph);
            if(maxEngines != 0)
            {
                hipdnn_plugin_sdk::throwIfNull(engineIds);
            }
            hipdnn_plugin_sdk::throwIfNull(numEngines);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            const auto ids = getInstance()->getAllEngineIds();
            *numEngines = static_cast<uint32_t>(ids.size());
            const auto copied = std::min<uint32_t>(maxEngines, *numEngines);
            for(uint32_t i = 0; i < copied; ++i)
            {
                engineIds[i] = ids[i];
            }

            LOG_API_SUCCESS(apiName, "numEngines=" << *numEngines);
        });
    }

    static hipdnnPluginStatus_t getEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                 int64_t engineId,
                                                 const hipdnnPluginConstData_t* opGraph,
                                                 hipdnnPluginConstData_t* engineDetails)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle) << ", engineId=" << engineId
                                << ", opGraph=" << static_cast<const void*>(opGraph)
                                << ", engineDetails=" << static_cast<void*>(engineDetails));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(opGraph);
            hipdnn_plugin_sdk::throwIfNull(engineDetails);

            if(engineId != K_KNOB_ENGINE_ID && engineId != K_CONTEXT_FAILS_ENGINE_ID)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                                               "Unknown engine id "
                                                                   + std::to_string(engineId));
            }

            flatbuffers::FlatBufferBuilder builder;
            std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>>
                knobOffsets;
            if(engineId == K_KNOB_ENGINE_ID)
            {
                // Plain min/max/step: the shim shows it as TILE_SIZE.
                knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createIntKnob(
                    builder, "test.tile_size", "Projectable tile size", 16, 0, 64, 8, {}));
                // Maps to BLOCK_SIZE by name, but explicit valid values have no cuDNN
                // representation, so the shim hides it.
                knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createIntKnob(
                    builder, "test.block_size", "Hidden block size", 8, 0, 0, 1, {8, 16}));
                // No cuDNN knob type matches this name.
                knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createIntKnob(
                    builder, "test.int_knob", "Unmappable integer knob", 50, 0, 100, 10, {}));
            }

            const std::vector<int32_t> behaviorNotes{
                static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION)};
            auto details = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetailsDirect(
                builder, engineId, &knobOffsets, &behaviorNotes);
            builder.Finish(details);
            auto serializedDetails = builder.Release();

            TestPluginMallocBuffer tempBuffer(std::malloc(serializedDetails.size()));
            hipdnn_plugin_sdk::throwIfNull(tempBuffer.get());
            std::memcpy(tempBuffer.get(), serializedDetails.data(), serializedDetails.size());

            engineDetails->ptr = tempBuffer.release();
            engineDetails->size = serializedDetails.size();

            LOG_API_SUCCESS(apiName, "engineDetails->ptr=" << engineDetails->ptr);
        });
    }

    static hipdnnPluginStatus_t
        createExecutionContext(hipdnnEnginePluginHandle_t handle,
                               const hipdnnPluginConstData_t* engineConfig,
                               const hipdnnPluginConstData_t* opGraph,
                               hipdnnEnginePluginExecutionContext_t* executionContext)
    {
        const auto status = hipdnn_plugin_sdk::tryCatch([&]() {
            hipdnn_plugin_sdk::throwIfNull(engineConfig);
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(
                engineConfig->ptr, engineConfig->size);
            if(config.engineId() == K_CONTEXT_FAILS_ENGINE_ID)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "Simulated execution context creation failure for testing");
            }
        });
        if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
        {
            return status;
        }
        return TestPluginBase::enginePluginCreateExecutionContext(
            handle, engineConfig, opGraph, executionContext);
    }
};

__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<CudnnShimV8Plugin>());
}

extern "C" {
HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    return TestPluginBase::pluginGetName(name);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    return TestPluginBase::pluginGetVersion(version);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetApiVersion(const char** version)
{
    return TestPluginBase::pluginGetApiVersion(version);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    return TestPluginBase::pluginGetType(type);
}

HIPDNN_TEST_PLUGIN_EXPORT void hipdnnPluginGetLastErrorString(const char** errorStr)
{
    TestPluginBase::pluginGetLastErrorString(errorStr);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)
{
    return TestPluginBase::pluginSetLoggingCallback(callback);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginSetLogLevel(hipdnnSeverity_t level)
{
    return TestPluginBase::pluginSetLogLevel(level);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
{
    return TestPluginBase::enginePluginGetAllEngineIds(engineIds, maxEngines, numEngines);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)
{
    return TestPluginBase::enginePluginCreate(handle);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)
{
    return TestPluginBase::enginePluginDestroy(handle);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle, hipStream_t stream)
{
    return TestPluginBase::enginePluginSetStream(handle, stream);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* opGraph,
                                             int64_t* engineIds,
                                             uint32_t maxEngines,
                                             uint32_t* numEngines)
{
    return CudnnShimV8Plugin::getApplicableEngineIds(
        handle, opGraph, engineIds, maxEngines, numEngines);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                       int64_t engineId,
                                       const hipdnnPluginConstData_t* opGraph,
                                       hipdnnPluginConstData_t* engineDetails)
{
    return CudnnShimV8Plugin::getEngineDetails(handle, engineId, opGraph, engineDetails);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnEnginePluginDestroyEngineDetails(
    hipdnnEnginePluginHandle_t handle, hipdnnPluginConstData_t* engineDetails)
{
    return TestPluginBase::enginePluginDestroyEngineDetails(handle, engineDetails);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                       const hipdnnPluginConstData_t* engineConfig,
                                       const hipdnnPluginConstData_t* opGraph,
                                       size_t* workspaceSize)
{
    if(engineConfig != nullptr && engineConfig->ptr != nullptr && engineConfig->size > 0)
    {
        const auto* bytes = static_cast<const uint8_t*>(engineConfig->ptr);
        gReceivedKnobs.emplace_back(bytes, bytes + engineConfig->size);
    }

    return TestPluginBase::enginePluginGetWorkspaceSize(
        handle, engineConfig, opGraph, workspaceSize);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(
        hipdnnEnginePluginHandle_t handle,
        hipdnnEnginePluginExecutionContext_t executionContext,
        size_t* workspaceSize)
{
    return TestPluginBase::enginePluginGetWorkspaceSize(handle, executionContext, workspaceSize);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecutionContext(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* engineConfig,
                                             const hipdnnPluginConstData_t* opGraph,
                                             hipdnnEnginePluginExecutionContext_t* executionContext)
{
    return CudnnShimV8Plugin::createExecutionContext(
        handle, engineConfig, opGraph, executionContext);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(
    hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t executionContext)
{
    return TestPluginBase::enginePluginDestroyExecutionContext(handle, executionContext);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnEnginePluginSerializeExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext,
    hipdnnPluginConstData_t* serializedContext)
{
    return TestPluginBase::enginePluginSerializeExecutionContext(
        handle, executionContext, serializedContext);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnEnginePluginDestroySerializedExecutionContext(
    hipdnnEnginePluginHandle_t handle, hipdnnPluginConstData_t* serializedContext)
{
    return TestPluginBase::enginePluginDestroySerializedExecutionContext(handle, serializedContext);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecutionContextFromSerialized(
        hipdnnEnginePluginHandle_t handle,
        const hipdnnPluginConstData_t* serializedContext,
        hipdnnEnginePluginExecutionContext_t* executionContext)
{
    return TestPluginBase::enginePluginCreateExecutionContextFromSerialized(
        handle, serializedContext, executionContext);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t executionContext,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                     uint32_t numDeviceBuffers)
{
    return TestPluginBase::enginePluginExecuteOpGraph(
        handle, executionContext, workspace, deviceBuffers, numDeviceBuffers);
}

HIPDNN_PLUGIN_EXPORT uint32_t hipdnnTestKnobsPluginGetReceivedKnobsCount()
{
    return static_cast<uint32_t>(gReceivedKnobs.size());
}

HIPDNN_PLUGIN_EXPORT const uint8_t* hipdnnTestKnobsPluginGetReceivedKnobsDataAt(uint32_t index)
{
    if(index >= gReceivedKnobs.size())
    {
        return nullptr;
    }
    return gReceivedKnobs[index].data();
}

HIPDNN_PLUGIN_EXPORT uint32_t hipdnnTestKnobsPluginGetReceivedKnobsSizeAt(uint32_t index)
{
    if(index >= gReceivedKnobs.size())
    {
        return 0;
    }
    return static_cast<uint32_t>(gReceivedKnobs[index].size());
}

HIPDNN_PLUGIN_EXPORT void hipdnnTestKnobsPluginResetReceivedKnobs()
{
    gReceivedKnobs.clear();
}
} // extern "C"
