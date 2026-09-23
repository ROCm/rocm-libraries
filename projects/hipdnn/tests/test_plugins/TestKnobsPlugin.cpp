// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>

#include <cstdint>
#include <vector>

// Thread-local storage for raw flatbuffer bytes of each received EngineConfig.
// Tests read these via the exported getCount/getDataAt/getSizeAt/reset C functions,
// then unpack to EngineConfigT for direct comparison.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static thread_local std::vector<std::vector<uint8_t>> gReceivedKnobs;

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class KnobsPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_KnobsPlugin";
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
        return hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    }

    uint32_t getNumEngines() const override
    {
        return 2;
    }

    uint32_t getNumApplicableEngines() const override
    {
        return 2;
    }

    // Override enginePluginGetAllEngineIds to return both engines
    static hipdnnPluginStatus_t
        getAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
    {
        LOG_API_ENTRY("engineIds=" << static_cast<void*>(engineIds) << ", maxEngines=" << maxEngines
                                   << ", numEngines=" << static_cast<void*>(numEngines));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            if(maxEngines != 0)
            {
                hipdnn_plugin_sdk::throwIfNull(engineIds);
            }
            hipdnn_plugin_sdk::throwIfNull(numEngines);

            *numEngines = 2;

            if(maxEngines >= 1)
            {
                engineIds[0] = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
            }
            if(maxEngines >= 2)
            {
                engineIds[1] = hipdnn_tests::plugin_constants::engineId<KnobsPluginEngineB>();
            }

            LOG_API_SUCCESS(apiName, "numEngines=" << *numEngines);
        });
    }

    // Override enginePluginGetApplicableEngineIds to return both engines
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

            *numEngines = 2;

            if(maxEngines >= 1)
            {
                engineIds[0] = hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
            }
            if(maxEngines >= 2)
            {
                engineIds[1] = hipdnn_tests::plugin_constants::engineId<KnobsPluginEngineB>();
            }

            LOG_API_SUCCESS(apiName, "numEngines=" << *numEngines);
        });
    }

    // Override enginePluginGetEngineDetails to return knobs
    static hipdnnPluginStatus_t getEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                 [[maybe_unused]] int64_t engineId,
                                                 const hipdnnPluginConstData_t* opGraph,
                                                 hipdnnPluginConstData_t* engineDetails)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", opGraph=" << static_cast<const void*>(opGraph)
                                << ", engineDetails=" << static_cast<void*>(engineDetails));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(opGraph);
            hipdnn_plugin_sdk::throwIfNull(engineDetails);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot get engine details");
            }

            flatbuffers::FlatBufferBuilder builder;

            // Create knobs vector using KnobFactory
            std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>>
                knobOffsets;

            if(engineId == hipdnn_tests::plugin_constants::engineId<KnobsPlugin>())
            {
                // Engine A (4 unique knobs + 1 shared knob)

                // Knob 1: Integer knob with min/max/step constraints
                knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createIntKnob(
                    builder,
                    "test.int_knob",
                    "Test integer knob with range 0-100",
                    50, // default value
                    0, // min
                    100, // max
                    10, // step
                    {})); // no explicit valid values

                // Knob 2: Float knob with min/max constraints
                knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createFloatKnob(
                    builder,
                    "test.float_knob",
                    "Test float knob with range 0.0-1.0",
                    0.5f, // default value
                    0.0f, // min
                    1.0f)); // max

                // Knob 3: String knob with valid_values constraint
                knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createStringKnob(
                    builder,
                    "test.string_knob",
                    "Test string knob with enum values",
                    "fast", // default value
                    {"fast", "accurate", "balanced"})); // valid values

                // Knob 4: Deprecated integer knob
                knobOffsets.push_back(
                    hipdnn_plugin_sdk::KnobFactory::createIntKnob(builder,
                                                                  "test.deprecated_knob",
                                                                  "Deprecated knob for testing",
                                                                  0, // default value
                                                                  0, // min
                                                                  10, // max
                                                                  1, // step
                                                                  {}, // no explicit valid values
                                                                  true)); // deprecated
            }
            else if(engineId == hipdnn_tests::plugin_constants::engineId<KnobsPluginEngineB>())
            {
                // Engine B (2 different unique knobs + 1 shared knob)

                // Knob 1: Int knob with specific value constraint.
                knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createIntKnob(
                    builder,
                    "test.engine_b.block_size",
                    "Block size for engine B (power of 2)",
                    16,
                    0,
                    0,
                    1,
                    {8, 16, 32, 64}));

                // Knob 2: String knob with specific values.
                knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createStringKnob(
                    builder,
                    "test.engine_b.algorithm",
                    "Algorithm selection for engine B",
                    "winograd",
                    {"direct", "winograd", "fft"}));
            }

            // New shared knob for both engines
            knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createIntKnob(
                builder,
                "test.shared.deterministic",
                "Enable deterministic execution (shared across engines)",
                0,
                0,
                1,
                1,
                {}));

            auto knobsVector = builder.CreateVector(knobOffsets);
            auto newEngineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(
                builder, engineId, knobsVector);
            builder.Finish(newEngineDetails);
            auto serializedDetails = builder.Release();

            TestPluginMallocBuffer tempBuffer(std::malloc(serializedDetails.size()));
            hipdnn_plugin_sdk::throwIfNull(tempBuffer.get());
            std::memcpy(tempBuffer.get(), serializedDetails.data(), serializedDetails.size());

            engineDetails->ptr = tempBuffer.release();
            engineDetails->size = serializedDetails.size();

            LOG_API_SUCCESS(apiName, "engineDetails->ptr=" << engineDetails->ptr);
        });
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<KnobsPlugin>());
}

// Custom API registration that overrides enginePluginGetEngineDetails
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

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
{
    return KnobsPlugin::getAllEngineIds(engineIds, maxEngines, numEngines);
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
    return KnobsPlugin::getApplicableEngineIds(handle, opGraph, engineIds, maxEngines, numEngines);
}

// Override to use KnobsPlugin::getEngineDetails
HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                       int64_t engineId,
                                       const hipdnnPluginConstData_t* opGraph,
                                       hipdnnPluginConstData_t* engineDetails)
{
    return KnobsPlugin::getEngineDetails(handle, engineId, opGraph, engineDetails);
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
    // Record raw flatbuffer bytes so tests can unpack and compare EngineConfigT directly.
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
    return TestPluginBase::enginePluginCreateExecutionContext(
        handle, engineConfig, opGraph, executionContext);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(
    hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t executionContext)
{
    return TestPluginBase::enginePluginDestroyExecutionContext(handle, executionContext);
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
