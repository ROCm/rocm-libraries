// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

class KnobConstraintValidationPlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_KnobConstraintValidationPlugin";
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
        return hipdnn_tests::plugin_constants::engineId<KnobConstraintValidationPlugin>();
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

            // Knob 1: Integer knob with min/max/step constraints
            knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createIntKnob(
                builder,
                "constraint.int_knob",
                "Integer knob for constraint validation testing",
                50, // default value
                0, // min
                100, // max
                5, // step
                {})); // no explicit valid values

            // Knob 2: Float knob with min/max constraints
            knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createFloatKnob(
                builder,
                "constraint.float_knob",
                "Float knob for constraint validation testing",
                5.0f, // default value
                0.0f, // min
                10.0f)); // max

            // Knob 3: String knob with valid_values constraint
            knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createStringKnob(
                builder,
                "constraint.string_knob",
                "String knob for constraint validation testing",
                "beta", // default value
                {"alpha", "beta", "gamma"})); // valid values

            auto knobsVector = builder.CreateVector(knobOffsets);
            auto newEngineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(
                builder, getInstance()->getEngineId(), knobsVector);
            builder.Finish(newEngineDetails);
            auto serializedDetails = builder.Release();

            auto* tempBuffer = new uint8_t[serializedDetails.size()];
            std::memcpy(tempBuffer, serializedDetails.data(), serializedDetails.size());

            engineDetails->ptr = tempBuffer;
            engineDetails->size = serializedDetails.size();

            LOG_API_SUCCESS(apiName, "engineDetails->ptr=" << engineDetails->ptr);
        });
    }

    // Override createExecutionContext to validate knob types
    static hipdnnPluginStatus_t
        createExecutionContext(hipdnnEnginePluginHandle_t handle,
                               const hipdnnPluginConstData_t* engineConfig,
                               const hipdnnPluginConstData_t* opGraph,
                               hipdnnEnginePluginExecutionContext_t* executionContext)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", engineConfig=" << static_cast<const void*>(engineConfig)
                                << ", opGraph=" << static_cast<const void*>(opGraph)
                                << ", executionContext=" << static_cast<void*>(executionContext));

        return hipdnn_plugin_sdk::tryCatch([&]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(engineConfig);
            hipdnn_plugin_sdk::throwIfNull(opGraph);
            hipdnn_plugin_sdk::throwIfNull(executionContext);

            // Deserialize engineConfig to access knob settings
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper configWrapper(
                engineConfig->ptr, engineConfig->size);

            // Validate knob types
            for(const auto& knobSetting : configWrapper.knobSettingWrappers())
            {
                auto knobId = knobSetting->knobId();
                auto valueType = knobSetting->valueType();

                // Match against known knobs and validate type
                if(knobId == "constraint.int_knob")
                {
                    if(valueType != hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue)
                    {
                        throw hipdnn_plugin_sdk::HipdnnPluginException(
                            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                            "Type mismatch for constraint.int_knob: expected IntValue");
                    }
                }
                else if(knobId == "constraint.float_knob")
                {
                    if(valueType != hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue)
                    {
                        throw hipdnn_plugin_sdk::HipdnnPluginException(
                            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                            "Type mismatch for constraint.float_knob: expected FloatValue");
                    }
                }
                else if(knobId == "constraint.string_knob")
                {
                    if(valueType != hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue)
                    {
                        throw hipdnn_plugin_sdk::HipdnnPluginException(
                            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                            "Type mismatch for constraint.string_knob: expected StringValue");
                    }
                }
                // Unknown knobs are silently ignored, real plugins might log a warning but were not
                // testing that
            }

            // Create execution context as normal using base class implementation
            return TestPluginBase::enginePluginCreateExecutionContext(
                handle, engineConfig, opGraph, executionContext);
        });
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<KnobConstraintValidationPlugin>());
}

// Custom API registration that overrides enginePluginGetEngineDetails and createExecutionContext
extern "C" {
hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    return TestPluginBase::pluginGetName(name);
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    return TestPluginBase::pluginGetVersion(version);
}

hipdnnPluginStatus_t hipdnnPluginGetApiVersion(const char** version)
{
    return TestPluginBase::pluginGetApiVersion(version);
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

// Override to use KnobConstraintValidationPlugin::getEngineDetails
hipdnnPluginStatus_t hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                        int64_t engineId,
                                                        const hipdnnPluginConstData_t* opGraph,
                                                        hipdnnPluginConstData_t* engineDetails)
{
    return KnobConstraintValidationPlugin::getEngineDetails(
        handle, engineId, opGraph, engineDetails);
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

// Override to use KnobConstraintValidationPlugin::createExecutionContext
hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecutionContext(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* engineConfig,
                                             const hipdnnPluginConstData_t* opGraph,
                                             hipdnnEnginePluginExecutionContext_t* executionContext)
{
    return KnobConstraintValidationPlugin::createExecutionContext(
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
