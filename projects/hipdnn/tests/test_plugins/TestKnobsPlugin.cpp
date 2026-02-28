// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>

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

    int64_t getEngineId() const override
    {
        return hipdnn_tests::plugin_constants::engineId<KnobsPlugin>();
    }

    uint32_t getNumEngines() const override
    {
        return 1;
    }

    uint32_t getNumApplicableEngines() const override
    {
        return 1;
    }

    // Override enginePluginGetEngineDetails to return knobs
    static hipdnnPluginStatus_t getEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                 int64_t engineId,
                                                 const hipdnnPluginConstData_t* opGraph,
                                                 hipdnnPluginConstData_t* engineDetails)
    {
        LOG_API_ENTRY("handle={:p}, engineId={}, opGraph={:p}, engineDetails={:p}",
                      static_cast<void*>(handle),
                      engineId,
                      static_cast<const void*>(opGraph),
                      static_cast<void*>(engineDetails));

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
            std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Knob>> knobOffsets;

            // Knob 1: Integer knob with min/max/step constraints
            knobOffsets.push_back(
                hipdnn_plugin_sdk::KnobFactory::createIntKnob(builder,
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

            auto knobsVector = builder.CreateVector(knobOffsets);
            auto newEngineDetails = hipdnn_data_sdk::data_objects::CreateEngineDetails(
                builder, getInstance()->getEngineId(), knobsVector);
            builder.Finish(newEngineDetails);
            auto serializedDetails = builder.Release();

            auto* tempBuffer = new uint8_t[serializedDetails.size()];
            std::memcpy(tempBuffer, serializedDetails.data(), serializedDetails.size());

            engineDetails->ptr = tempBuffer;
            engineDetails->size = serializedDetails.size();

            LOG_API_SUCCESS(apiName, "engineDetails->ptr={:p}", engineDetails->ptr);
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
hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    return TestPluginBase::pluginGetName(name);
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    return TestPluginBase::pluginGetVersion(version);
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    return TestPluginBase::pluginGetType(type);
}

void hipdnnPluginGetLastErrorString(const char** errorStr)
{
    TestPluginBase::pluginGetLastErrorString(errorStr);
}

hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)
{
    return TestPluginBase::pluginSetLoggingCallback(callback);
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
{
    return TestPluginBase::enginePluginGetAllEngineIds(engineIds, maxEngines, numEngines);
}

hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)
{
    return TestPluginBase::enginePluginCreate(handle);
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)
{
    return TestPluginBase::enginePluginDestroy(handle);
}

hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                                                 hipStream_t stream)
{
    return TestPluginBase::enginePluginSetStream(handle, stream);
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* opGraph,
                                             int64_t* engineIds,
                                             uint32_t maxEngines,
                                             uint32_t* numEngines)
{
    return TestPluginBase::enginePluginGetApplicableEngineIds(
        handle, opGraph, engineIds, maxEngines, numEngines);
}

// Override to use KnobsPlugin::getEngineDetails
hipdnnPluginStatus_t hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                        int64_t engineId,
                                                        const hipdnnPluginConstData_t* opGraph,
                                                        hipdnnPluginConstData_t* engineDetails)
{
    return KnobsPlugin::getEngineDetails(handle, engineId, opGraph, engineDetails);
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                            hipdnnPluginConstData_t* engineDetails)
{
    return TestPluginBase::enginePluginDestroyEngineDetails(handle, engineDetails);
}

hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                                        const hipdnnPluginConstData_t* engineConfig,
                                                        const hipdnnPluginConstData_t* opGraph,
                                                        size_t* workspaceSize)
{
    return TestPluginBase::enginePluginGetWorkspaceSize(
        handle, engineConfig, opGraph, workspaceSize);
}

hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext,
    size_t* workspaceSize)
{
    return TestPluginBase::enginePluginGetWorkspaceSize(handle, executionContext, workspaceSize);
}

hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecutionContext(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* engineConfig,
                                             const hipdnnPluginConstData_t* opGraph,
                                             hipdnnEnginePluginExecutionContext_t* executionContext)
{
    return TestPluginBase::enginePluginCreateExecutionContext(
        handle, engineConfig, opGraph, executionContext);
}

hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyExecutionContext(hipdnnEnginePluginHandle_t handle,
                                              hipdnnEnginePluginExecutionContext_t executionContext)
{
    return TestPluginBase::enginePluginDestroyExecutionContext(handle, executionContext);
}

hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t executionContext,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                     uint32_t numDeviceBuffers)
{
    return TestPluginBase::enginePluginExecuteOpGraph(
        handle, executionContext, workspace, deviceBuffers, numDeviceBuffers);
}
} // extern "C"
