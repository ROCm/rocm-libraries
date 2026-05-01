// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file EnginePluginImpl.inl
 * @brief Inline implementation file for engine plugin C API entry points.
 *
 * This file provides the complete implementation of all required C API entry
 * points for an engine plugin. Include this file in your plugin's .cpp file
 * after defining the required macros and types.
 *
 * ## Required Macros (define before including this file)
 *
 * - HIPDNN_PLUGIN_NAME: String literal for the plugin name (e.g., "my_plugin")
 * - HIPDNN_PLUGIN_VERSION: String literal for the plugin version (e.g., "1.0.0")
 *   - This is typically defined in version.h for your project.
 * - HIPDNN_PLUGIN_CONTAINER_TYPE: The container class type
 * - HIPDNN_PLUGIN_HANDLE_TYPE: The handle struct type (inherits from HipdnnEnginePluginHandle)
 * - HIPDNN_PLUGIN_CONTEXT_TYPE: The context struct type (inherits from HipdnnEnginePluginExecutionContext)
 *
 * ## Required Types (define before including this file)
 *
 * 1. Handle type (HIPDNN_PLUGIN_HANDLE_TYPE):
 *    - Must inherit from: HipdnnEnginePluginHandle
 *    - Must have: std::shared_ptr<HIPDNN_PLUGIN_CONTAINER_TYPE> container
 *    - Must have: void setStream(hipStream_t)
 *    - Must have: EngineManager<...>& getEngineManager()
 *    - Must have: void removeEngineDetailsDetachedBuffer(const void*)
 *
 * 2. Context type (HIPDNN_PLUGIN_CONTEXT_TYPE):
 *    - Must inherit from: HipdnnEnginePluginExecutionContext
 *    - Must inherit from: ExecutionContextBase<THandle, TSettings>
 *    - Must have: IPlan<THandle>& plan()
 *
 * 3. Container type (HIPDNN_PLUGIN_CONTAINER_TYPE):
 *    - Must have: EngineManager<...>& getEngineManager()
 *    - Must have: static uint32_t copyEngineIds(int64_t*, uint32_t, uint32_t&)
 *
 * ## Usage Example
 *
 * ```cpp
 * // MiopenPluginPublic.cpp
 * #include "HipdnnMiopenTypes.hpp"
 * #include "MiopenContainer.hpp"
 * #include "version.h"
 *
 * using namespace miopen_plugin;
 *
 * #define HIPDNN_PLUGIN_NAME "miopen_provider_plugin"
 * #define HIPDNN_PLUGIN_VERSION MIOPEN_PROVIDER_VERSION_STRING // from version.h
 * #define HIPDNN_PLUGIN_CONTAINER_TYPE MiopenContainer
 * #define HIPDNN_PLUGIN_HANDLE_TYPE HipdnnMiopenHandle
 * #define HIPDNN_PLUGIN_CONTEXT_TYPE HipdnnMiopenContext
 *
 * #include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
 * ```
 */

#ifndef HIPDNN_PLUGIN_NAME
#error "HIPDNN_PLUGIN_NAME must be defined before including EnginePluginImpl.inl"
#endif

#ifndef HIPDNN_PLUGIN_VERSION
#error "HIPDNN_PLUGIN_VERSION must be defined before including EnginePluginImpl.inl"
#endif

#ifndef HIPDNN_PLUGIN_CONTAINER_TYPE
#error "HIPDNN_PLUGIN_CONTAINER_TYPE must be defined before including EnginePluginImpl.inl"
#endif

#ifndef HIPDNN_PLUGIN_HANDLE_TYPE
#error "HIPDNN_PLUGIN_HANDLE_TYPE must be defined before including EnginePluginImpl.inl"
#endif

#ifndef HIPDNN_PLUGIN_CONTEXT_TYPE
#error "HIPDNN_PLUGIN_CONTEXT_TYPE must be defined before including EnginePluginImpl.inl"
#endif

#include <memory>
#include <mutex>

#include <hip/hip_runtime.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/EnginePluginTypeTraits.hpp>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/PluginHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>
#include <hipdnn_plugin_sdk/SharedContainerManager.hpp>

// Compile-time validation of container type
namespace
{
// NOLINTBEGIN(readability-identifier-naming)
[[maybe_unused]] constexpr bool CONTAINER_TYPE_VALIDATION
    = (hipdnn_plugin_sdk::validateContainerType<HIPDNN_PLUGIN_CONTAINER_TYPE>(), true);

[[maybe_unused]] constexpr bool HANDLE_TYPE_VALIDATION
    = (hipdnn_plugin_sdk::validateHandleType<HIPDNN_PLUGIN_HANDLE_TYPE,
                                             HIPDNN_PLUGIN_CONTAINER_TYPE>(),
       true);
// NOLINTEND(readability-identifier-naming)
} // namespace

// Static plugin metadata
// NOLINTBEGIN(readability-identifier-naming)
static const char* pluginName = HIPDNN_PLUGIN_NAME;
static const char* pluginVersion = HIPDNN_PLUGIN_VERSION;
// NOLINTEND(readability-identifier-naming)

// Thread-local error string storage
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char
    hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]
    = "";

// Shared container manager for container lifecycle
// NOLINTNEXTLINE(readability-identifier-naming)
static hipdnn_plugin_sdk::SharedContainerManager<HIPDNN_PLUGIN_CONTAINER_TYPE> containerManager;

extern "C" {

// =============================================================================
// Base plugin API functions
// =============================================================================

hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    LOG_API_ENTRY("name_ptr=" << static_cast<void*>(name));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(name);

        *name = pluginName;

        LOG_API_SUCCESS(apiName, "pluginName=" << static_cast<void*>(name));
    });
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    LOG_API_ENTRY("versionPtr=" << static_cast<void*>(version));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(version);

        *version = pluginVersion;

        LOG_API_SUCCESS(apiName, "version=" << static_cast<void*>(version));
    });
}

// TODO: Turn this into an error in the future to make this required
#ifdef HIPDNN_PLUGIN_API_VERSION
hipdnnPluginStatus_t hipdnnPluginGetApiVersion(const char** version)
{
    LOG_API_ENTRY("versionPtr=" << static_cast<void*>(version));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(version);

        *version = HIPDNN_PLUGIN_API_VERSION;

        LOG_API_SUCCESS(apiName, "version=" << static_cast<void*>(version));
    });
}
#endif

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    LOG_API_ENTRY("typePtr=" << static_cast<void*>(type));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(type);

        *type = HIPDNN_PLUGIN_TYPE_ENGINE;

        LOG_API_SUCCESS(apiName, "type=" << toString(*type));
    });
}

void hipdnnPluginGetLastErrorString(const char** errorStr)
{
    LOG_API_ENTRY("errorStrPtr=" << static_cast<void*>(errorStr));

    hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(errorStr);

        *errorStr = hipdnn_plugin_sdk::PluginLastErrorManager::getLastError();

        LOG_API_SUCCESS(apiName, "errorStr=" << static_cast<void*>(errorStr));
    });
}

hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)
{
    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(callback);
        hipdnn_plugin_sdk::logging::initializeCallbackLogging(pluginName, callback);
        LOG_API_SUCCESS(apiName, "");
    });
}

hipdnnPluginStatus_t hipdnnPluginSetLogLevel(hipdnnSeverity_t level)
{
    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::logging::setLogLevel(level);
        LOG_API_SUCCESS(apiName, "level=" << level);
    });
}

// =============================================================================
// Engine plugin API functions
// =============================================================================

hipdnnPluginStatus_t
    hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
{
    LOG_API_ENTRY("engineIds=" << static_cast<void*>(engineIds) << ", maxEngines=" << maxEngines
                               << ", numEngines=" << static_cast<void*>(numEngines));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        if(maxEngines != 0)
        {
            hipdnn_plugin_sdk::throwIfNull(engineIds);
        }
        hipdnn_plugin_sdk::throwIfNull(numEngines);

        auto totalEngines
            = HIPDNN_PLUGIN_CONTAINER_TYPE::copyEngineIds(engineIds, maxEngines, *numEngines);

        LOG_API_SUCCESS(apiName, "numEngines=" << *numEngines << " totalEngines=" << totalEngines);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)
{
    LOG_API_ENTRY("handle_ptr=" << static_cast<void*>(handle));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(handle);

        auto* newHandle = new HIPDNN_PLUGIN_HANDLE_TYPE();
        newHandle->container = containerManager.getOrCreate();
        *handle = static_cast<hipdnnEnginePluginHandle_t>(newHandle);

        LOG_API_SUCCESS(apiName, "createdHandle=" << static_cast<void*>(*handle));
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(handle);

        auto* typedHandle = static_cast<HIPDNN_PLUGIN_HANDLE_TYPE*>(handle);
        delete typedHandle;

        LOG_API_SUCCESS(apiName, "");
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                                                 hipStream_t stream)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", stream_id=" << static_cast<void*>(stream));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(handle);

        auto* typedHandle = static_cast<HIPDNN_PLUGIN_HANDLE_TYPE*>(handle);
        typedHandle->setStream(stream);

        LOG_API_SUCCESS(apiName, "");
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* opGraph,
                                             int64_t* engineIds,
                                             uint32_t maxEngines,
                                             uint32_t* numEngines)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", opGraph=" << static_cast<const void*>(opGraph) << ", engineIds="
                            << static_cast<void*>(engineIds) << ", maxEngines=" << maxEngines
                            << ", numEngines=" << static_cast<void*>(numEngines));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(handle);
        hipdnn_plugin_sdk::throwIfNull(opGraph);
        if(maxEngines != 0)
        {
            hipdnn_plugin_sdk::throwIfNull(engineIds);
        }
        hipdnn_plugin_sdk::throwIfNull(numEngines);

        auto* typedHandle = static_cast<HIPDNN_PLUGIN_HANDLE_TYPE*>(handle);
        auto& engineManager = typedHandle->getEngineManager();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper opGraphWrapper(opGraph->ptr,
                                                                           opGraph->size);

        auto applicableEngines = engineManager.getApplicableEngineIds(*typedHandle, opGraphWrapper);

        *numEngines = 0;
        for(auto& engineId : applicableEngines)
        {
            if(*numEngines == maxEngines)
            {
                *numEngines = static_cast<uint32_t>(applicableEngines.size());
                HIPDNN_SDK_LOG_INFO("Maximum number of engines reached ("
                                    << maxEngines << "), ignoring additional "
                                    << "engines, numEngines count: " << *numEngines);
                break;
            }

            engineIds[*numEngines] = engineId;
            (*numEngines)++;
        }

        LOG_API_SUCCESS(apiName, "numEngines=" << *numEngines);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
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

        auto* typedHandle = static_cast<HIPDNN_PLUGIN_HANDLE_TYPE*>(handle);
        auto& engineManager = typedHandle->getEngineManager();
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper opGraphWrapper(opGraph->ptr,
                                                                           opGraph->size);

        engineManager.getEngineDetails(*typedHandle, opGraphWrapper, engineId, *engineDetails);

        LOG_API_SUCCESS(apiName, "engineDetails->ptr=" << engineDetails->ptr);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                            hipdnnPluginConstData_t* engineDetails)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", engineDetails=" << static_cast<void*>(engineDetails));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(handle);
        hipdnn_plugin_sdk::throwIfNull(engineDetails);
        hipdnn_plugin_sdk::throwIfNull(engineDetails->ptr);

        auto* typedHandle = static_cast<HIPDNN_PLUGIN_HANDLE_TYPE*>(handle);
        typedHandle->removeEngineDetailsDetachedBuffer(engineDetails->ptr);

        LOG_API_SUCCESS(apiName, "engineDetails->ptr=" << engineDetails->ptr);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
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

        auto* typedHandle = static_cast<HIPDNN_PLUGIN_HANDLE_TYPE*>(handle);
        auto& engineManager = typedHandle->getEngineManager();

        hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper engineConfigWrapper(
            engineConfig->ptr, engineConfig->size);
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper opGraphWrapper(opGraph->ptr,
                                                                           opGraph->size);
        *workspaceSize
            = engineManager.getMaxWorkspaceSize(*typedHandle, opGraphWrapper, engineConfigWrapper);

        LOG_API_SUCCESS(apiName, "workspaceSize=" << *workspaceSize);
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecutionContext(hipdnnEnginePluginHandle_t handle,
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

        auto* typedHandle = static_cast<HIPDNN_PLUGIN_HANDLE_TYPE*>(handle);

        hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper opGraphWrapper(opGraph->ptr,
                                                                           opGraph->size);
        hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper engineConfigWrapper(
            engineConfig->ptr, engineConfig->size);

        auto& engineManager = typedHandle->getEngineManager();

        auto* context = new HIPDNN_PLUGIN_CONTEXT_TYPE;

        try
        {
            engineManager.initializeExecutionContext(
                *typedHandle, opGraphWrapper, engineConfigWrapper, *context);
        }
        catch(...)
        {
            delete context;
            throw;
        }

        *executionContext = static_cast<hipdnnEnginePluginExecutionContext_t>(context);

        LOG_API_SUCCESS(apiName,
                        "created_execution_context=" << static_cast<void*>(*executionContext));
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyExecutionContext(hipdnnEnginePluginHandle_t handle,
                                              hipdnnEnginePluginExecutionContext_t executionContext)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", executionContext=" << static_cast<void*>(executionContext));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(handle);
        hipdnn_plugin_sdk::throwIfNull(executionContext);

        auto* typedContext = static_cast<HIPDNN_PLUGIN_CONTEXT_TYPE*>(executionContext);
        delete typedContext;

        LOG_API_SUCCESS(apiName, "destroyed executionContext");
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext,
    size_t* workspaceSize)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", executionContext=" << static_cast<const void*>(executionContext)
                            << ", workspaceSize=" << static_cast<void*>(workspaceSize));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(handle);
        hipdnn_plugin_sdk::throwIfNull(executionContext);
        hipdnn_plugin_sdk::throwIfNull(workspaceSize);

        auto* typedHandle = static_cast<HIPDNN_PLUGIN_HANDLE_TYPE*>(handle);
        auto* typedContext = static_cast<HIPDNN_PLUGIN_CONTEXT_TYPE*>(executionContext);
        *workspaceSize = typedContext->plan().getWorkspaceSize(*typedHandle);

        LOG_API_SUCCESS(apiName, "workspaceSize=" << *workspaceSize);
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t executionContext,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                     uint32_t numDeviceBuffers)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle) << ", executionContext="
                            << static_cast<void*>(executionContext) << ", workspace=" << workspace
                            << ", deviceBuffers=" << static_cast<const void*>(deviceBuffers)
                            << ", numDeviceBuffers=" << numDeviceBuffers);

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::throwIfNull(handle);
        hipdnn_plugin_sdk::throwIfNull(executionContext);
        hipdnn_plugin_sdk::throwIfNull(deviceBuffers);

        auto* typedHandle = static_cast<HIPDNN_PLUGIN_HANDLE_TYPE*>(handle);
        auto* typedContext = static_cast<HIPDNN_PLUGIN_CONTEXT_TYPE*>(executionContext);
        typedContext->plan().execute(*typedHandle, deviceBuffers, numDeviceBuffers, workspace);

        LOG_API_SUCCESS(apiName, "executed graph");
    });
}

} // extern "C"
