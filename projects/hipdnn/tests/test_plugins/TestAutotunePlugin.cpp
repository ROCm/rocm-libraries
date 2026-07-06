// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestPluginCommon.hpp"
#include "TestPluginEngineIdMap.hpp"

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

// NOLINTNEXTLINE
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

// Workspace sizes for AutotunePluginEngineWorkspaceGrows: the pre-compile
// estimate fits a modest budget, but the compiled (execution-plan) workspace is
// larger. A test sets a budget between the two so EXHAUSTIVE priming is skipped
// for this engine while the normal benchmark still runs.
namespace
{
constexpr size_t WORKSPACE_DEFAULT_SIZE = 1024;
constexpr size_t WORKSPACE_LARGE_COMPILED_SIZE = 8192;
} // namespace

class AutotunePlugin : public TestPluginBase
{
public:
    const char* getPluginName() const override
    {
        return "test_AutotunePlugin";
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
        return hipdnn_tests::plugin_constants::engineId<AutotunePlugin>();
    }

    uint32_t getNumEngines() const override
    {
        return 6;
    }

    uint32_t getNumApplicableEngines() const override
    {
        return 6;
    }

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

            constexpr uint32_t TOTAL_ENGINES = 6;
            // When maxEngines=0, return total count for discovery; otherwise return actual count
            *numEngines = (maxEngines == 0) ? TOTAL_ENGINES : std::min(maxEngines, TOTAL_ENGINES);

            if(maxEngines >= 1)
            {
                engineIds[0] = hipdnn_tests::plugin_constants::engineId<AutotunePlugin>();
            }
            if(maxEngines >= 2)
            {
                engineIds[1] = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineB>();
            }
            if(maxEngines >= 3)
            {
                engineIds[2] = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineC>();
            }
            if(maxEngines >= 4)
            {
                engineIds[3]
                    = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineFails>();
            }
            if(maxEngines >= 5)
            {
                engineIds[4] = hipdnn_tests::plugin_constants::engineId<
                    AutotunePluginEnginePrimingOnlyFails>();
            }
            if(maxEngines >= 6)
            {
                engineIds[5] = hipdnn_tests::plugin_constants::engineId<
                    AutotunePluginEngineWorkspaceGrows>();
            }

            LOG_API_SUCCESS(apiName, "numEngines=" << *numEngines);
        });
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

            constexpr uint32_t TOTAL_ENGINES = 6;
            // When maxEngines=0, return total count for discovery; otherwise return actual count
            *numEngines = (maxEngines == 0) ? TOTAL_ENGINES : std::min(maxEngines, TOTAL_ENGINES);

            if(maxEngines >= 1)
            {
                engineIds[0] = hipdnn_tests::plugin_constants::engineId<AutotunePlugin>();
            }
            if(maxEngines >= 2)
            {
                engineIds[1] = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineB>();
            }
            if(maxEngines >= 3)
            {
                engineIds[2] = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineC>();
            }
            if(maxEngines >= 4)
            {
                engineIds[3]
                    = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineFails>();
            }
            if(maxEngines >= 5)
            {
                engineIds[4] = hipdnn_tests::plugin_constants::engineId<
                    AutotunePluginEnginePrimingOnlyFails>();
            }
            if(maxEngines >= 6)
            {
                engineIds[5] = hipdnn_tests::plugin_constants::engineId<
                    AutotunePluginEngineWorkspaceGrows>();
            }

            LOG_API_SUCCESS(apiName, "numEngines=" << *numEngines);
        });
    }

    // Custom createExecutionContext that parses knob settings to detect priming mode.
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

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(engineConfig);
            hipdnn_plugin_sdk::throwIfNull(opGraph);
            hipdnn_plugin_sdk::throwIfNull(executionContext);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot create execution context");
            }

            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper opGraphWrapper(
                opGraph->ptr, opGraph->size);
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper
                engineConfigWrapper(engineConfig->ptr, engineConfig->size);

            auto context = std::make_unique<HipdnnEnginePluginExecutionContext>();
            context->engineId = engineConfigWrapper.engineId();

            // Parse knob settings to detect benchmarking mode (global.benchmarking = 1)
            if(engineConfigWrapper.hasKnobSetting("global.benchmarking"))
            {
                const auto& knobSetting
                    = engineConfigWrapper.getKnobSettingByName("global.benchmarking");
                if(knobSetting.valueType()
                   == hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue)
                {
                    const auto& intValue
                        = knobSetting.valueAs<hipdnn_flatbuffers_sdk::data_objects::IntValue>();
                    if(intValue.value() == 1)
                    {
                        context->hasBenchmarkingKnobEnabled = true;
                    }
                }
            }

            *executionContext = context.release();

            LOG_API_SUCCESS(apiName,
                            "createdExecutionContext=" << static_cast<void*>(*executionContext));
        });
    }

    // AutotunePluginEngineFails: fails executeGraph() UNCONDITIONALLY so both priming
    // AND benchmark fail and succeeded==false holds.
    // AutotunePluginEnginePrimingOnlyFails: fails only during priming (hasBenchmarkingKnobEnabled=true),
    // succeeds during normal benchmark. Tests the RFC case where "succeeded may be true".
    static hipdnnPluginStatus_t
        executeOpGraph(hipdnnEnginePluginHandle_t handle,
                       hipdnnEnginePluginExecutionContext_t executionContext,
                       void* workspace,
                       const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                       uint32_t numDeviceBuffers)
    {
        if(executionContext != nullptr)
        {
            const auto* ctx = static_cast<HipdnnEnginePluginExecutionContext*>(executionContext);

            // AutotunePluginEngineFails: fails unconditionally
            if(ctx->engineId
               == hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineFails>())
            {
                return hipdnn_plugin_sdk::tryCatch([]() {
                    throw hipdnn_plugin_sdk::HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                        "AutotunePluginEngineFails: executeGraph fails unconditionally");
                });
            }

            // AutotunePluginEnginePrimingOnlyFails: fails only when benchmarking knob is enabled
            if(ctx->engineId
               == hipdnn_tests::plugin_constants::engineId<AutotunePluginEnginePrimingOnlyFails>())
            {
                if(ctx->hasBenchmarkingKnobEnabled)
                {
                    return hipdnn_plugin_sdk::tryCatch([]() {
                        throw hipdnn_plugin_sdk::HipdnnPluginException(
                            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                            "AutotunePluginEnginePrimingOnlyFails: priming execution failed");
                    });
                }
                // Normal execution succeeds - fall through to base implementation
            }
        }
        return TestPluginBase::enginePluginExecuteOpGraph(
            handle, executionContext, workspace, deviceBuffers, numDeviceBuffers);
    }

    static hipdnnPluginStatus_t getWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                                 const hipdnnPluginConstData_t* engineConfig,
                                                 const hipdnnPluginConstData_t* opGraph,
                                                 size_t* workspaceSize)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", engineConfig=" << static_cast<const void*>(engineConfig)
                                << ", opGraph=" << static_cast<const void*>(opGraph)
                                << ", workspaceSize=" << static_cast<void*>(workspaceSize));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(engineConfig);
            hipdnn_plugin_sdk::throwIfNull(opGraph);
            hipdnn_plugin_sdk::throwIfNull(workspaceSize);

            *workspaceSize = WORKSPACE_DEFAULT_SIZE;

            LOG_API_SUCCESS(apiName, "workspaceSize=" << *workspaceSize);
        });
    }

    // Compiled (execution-plan) workspace size, keyed on the engine ID and benchmarking
    // knob state stored im the execution context, allowing each engines to produce different
    // behaviors needed for the varioius test scenarios.
    static hipdnnPluginStatus_t
        getWorkspaceSizeFromExecutionContext(hipdnnEnginePluginHandle_t handle,
                                             hipdnnEnginePluginExecutionContext_t executionContext,
                                             size_t* workspaceSize)
    {
        LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                                << ", executionContext=" << static_cast<void*>(executionContext)
                                << ", workspaceSize=" << static_cast<void*>(workspaceSize));

        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin_sdk::throwIfNull(handle);
            hipdnn_plugin_sdk::throwIfNull(executionContext);
            hipdnn_plugin_sdk::throwIfNull(workspaceSize);
            hipdnn_plugin_sdk::throwIfNull(getInstance());

            const auto* ctx = static_cast<HipdnnEnginePluginExecutionContext*>(executionContext);

            if(ctx->engineId
               == hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineWorkspaceGrows>())
            {
                *workspaceSize = ctx->hasBenchmarkingKnobEnabled ? WORKSPACE_LARGE_COMPILED_SIZE
                                                                 : WORKSPACE_DEFAULT_SIZE;
            }
            else
            {
                *workspaceSize = WORKSPACE_DEFAULT_SIZE;
            }

            LOG_API_SUCCESS(apiName, "workspaceSize=" << *workspaceSize);
        });
    }

    // Override to return knobs that support the autotune workflow
    static hipdnnPluginStatus_t getEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                 int64_t engineId,
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

            std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>>
                knobOffsets;

            const auto engineA = hipdnn_tests::plugin_constants::engineId<AutotunePlugin>();
            const auto engineB = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineB>();
            const auto engineFails
                = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineFails>();
            const auto enginePrimingOnlyFails
                = hipdnn_tests::plugin_constants::engineId<AutotunePluginEnginePrimingOnlyFails>();
            const auto engineWorkspaceGrows
                = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineWorkspaceGrows>();

            if(engineId == engineFails || engineId == enginePrimingOnlyFails
               || engineId == engineWorkspaceGrows)
            {
                // These engines need the benchmarking knob so they are selected for
                // EXHAUSTIVE priming. EngineFails fails unconditionally;
                // EnginePrimingOnlyFails fails only during priming (testing RFC's
                // "succeeded may be true" case); EngineWorkspaceGrows reports a
                // larger workspace size that can be too large for the budget so
                // priming can be  skipped (not failed) while the real plan still benchmarks.
                knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createIntKnob(
                    builder,
                    "global.benchmarking",
                    "Enable benchmarking mode for cache priming",
                    0,
                    0,
                    1,
                    1,
                    {}));
            }
            else if(engineId == engineA)
            {
                // Engine A: global.benchmarking + test.autotune_variant knobs
                knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createIntKnob(
                    builder,
                    "global.benchmarking",
                    "Enable benchmarking mode for cache priming",
                    0, // default value
                    0, // min
                    1, // max
                    1, // step
                    {})); // no explicit valid values

                knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createIntKnob(
                    builder,
                    "test.autotune_variant",
                    "Test variant knob for autotune testing",
                    0, // default value
                    0, // min
                    3, // max
                    1, // step
                    {})); // no explicit valid values
            }
            else if(engineId == engineB)
            {
                // Engine B: algorithm string knob
                knobOffsets.push_back(hipdnn_plugin_sdk::KnobFactory::createStringKnob(
                    builder,
                    "test.engine_b.algorithm",
                    "Algorithm selection for engine B",
                    "direct",
                    {"direct", "winograd", "fft"}));
            }
            // Engine C: no knobs (empty vector)

            auto knobsVector = builder.CreateVector(knobOffsets);
            auto newEngineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(
                builder, engineId, knobsVector);
            builder.Finish(newEngineDetails);
            auto serializedDetails = builder.Release();

            auto* tempBuffer = static_cast<uint8_t*>(std::malloc(serializedDetails.size()));
            std::memcpy(tempBuffer, serializedDetails.data(), serializedDetails.size());

            engineDetails->ptr = tempBuffer;
            engineDetails->size = serializedDetails.size();

            LOG_API_SUCCESS(apiName, "engineDetails->ptr=" << engineDetails->ptr);
        });
    }
};

// Initialize plugin instance on load
__attribute__((constructor)) static void initializePlugin()
{
    TestPluginBase::setInstance(std::make_unique<AutotunePlugin>());
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

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t hipdnnPluginSetLogLevel(hipdnnSeverity_t level)
{
    return TestPluginBase::pluginSetLogLevel(level);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
{
    return AutotunePlugin::getAllEngineIds(engineIds, maxEngines, numEngines);
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
    return AutotunePlugin::getApplicableEngineIds(
        handle, opGraph, engineIds, maxEngines, numEngines);
}

// Override to use AutotunePlugin::getEngineDetails (returns knobs)
HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                       int64_t engineId,
                                       const hipdnnPluginConstData_t* opGraph,
                                       hipdnnPluginConstData_t* engineDetails)
{
    return AutotunePlugin::getEngineDetails(handle, engineId, opGraph, engineDetails);
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
    return AutotunePlugin::getWorkspaceSize(handle, engineConfig, opGraph, workspaceSize);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(
        hipdnnEnginePluginHandle_t handle,
        hipdnnEnginePluginExecutionContext_t executionContext,
        size_t* workspaceSize)
{
    return AutotunePlugin::getWorkspaceSizeFromExecutionContext(
        handle, executionContext, workspaceSize);
}

HIPDNN_TEST_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecutionContext(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* engineConfig,
                                             const hipdnnPluginConstData_t* opGraph,
                                             hipdnnEnginePluginExecutionContext_t* executionContext)
{
    return AutotunePlugin::createExecutionContext(handle, engineConfig, opGraph, executionContext);
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
    return AutotunePlugin::executeOpGraph(
        handle, executionContext, workspace, deviceBuffers, numDeviceBuffers);
}
} // extern "C"
