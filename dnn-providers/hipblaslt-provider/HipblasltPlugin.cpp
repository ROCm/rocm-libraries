// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/PluginDataTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "EngineManager.hpp"
#include "HipblasltContainer.hpp"
#include "HipblasltHandleFactory.hpp"
#include "HipblasltPlugin.hpp"
#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "version.h"

static const char* pluginName = "hipblaslt_plugin";
static const char* pluginVersion = HIPBLASLT_PROVIDER_VERSION_STRING;

using namespace hipdnn_plugin_sdk;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipblaslt_plugin;

// NOLINTNEXTLINE
thread_local char PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

std::weak_ptr<HipblasltContainer> hipblasltContainerLifecyclePtr;

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetNameImpl(const char** name)
{
    LOG_API_ENTRY("name_ptr=" << static_cast<void*>(name));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(name);

        *name = pluginName;

        LOG_API_SUCCESS(apiName, "pluginName=" << static_cast<void*>(name));
    });
}

hipdnnPluginStatus_t hipdnnPluginGetVersionImpl(const char** version)
{
    LOG_API_ENTRY("versionPtr=" << static_cast<void*>(version));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(version);

        *version = pluginVersion;

        LOG_API_SUCCESS(apiName, "version=" << static_cast<void*>(version));
    });
}

hipdnnPluginStatus_t hipdnnPluginGetTypeImpl(hipdnnPluginType_t* type)
{
    LOG_API_ENTRY("typePtr=" << static_cast<void*>(type));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(type);

        *type = HIPDNN_PLUGIN_TYPE_ENGINE;

        LOG_API_SUCCESS(apiName, "type=" << toString(*type));
    });
}

void hipdnnPluginGetLastErrorStringImpl(const char** errorStr)
{
    LOG_API_ENTRY("errorStrPtr=" << static_cast<void*>(errorStr));

    hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(errorStr);

        *errorStr = PluginLastErrorManager::getLastError();

        LOG_API_SUCCESS(apiName, "errorStr=" << static_cast<void*>(errorStr));
    });
}

// Once plugins are loaded via plugin manager then logging will work for them
hipdnnPluginStatus_t hipdnnPluginSetLoggingCallbackImpl(hipdnnCallback_t callback)
{
    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(callback);
        hipdnn_plugin_sdk::logging::initializeCallbackLogging(pluginName, callback);
        LOG_API_SUCCESS(apiName, "");
    });
}

hipdnnPluginStatus_t hipdnnPluginSetLogLevelImpl(hipdnnSeverity_t level)
{
    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        hipdnn_plugin_sdk::logging::setLogLevel(level);
        LOG_API_SUCCESS(apiName, "level=" << level);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetAllEngineIdsImpl(int64_t* engineIds,
                                                           uint32_t maxEngines,
                                                           uint32_t* numEngines)
{
    LOG_API_ENTRY("engineIds=" << static_cast<void*>(engineIds) << ", maxEngines=" << maxEngines
                               << ", numEngines=" << static_cast<void*>(numEngines));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        if(maxEngines != 0)
        {
            throwIfNull(engineIds);
        }
        throwIfNull(numEngines);

        auto allEngineIds = std::vector<int64_t>({hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID});
        if(allEngineIds.size() > std::numeric_limits<uint32_t>::max())
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                        "Number of engines exceeds maximum uint32_t value.");
        }

        if(maxEngines == 0)
        {
            *numEngines = static_cast<uint32_t>(allEngineIds.size());
        }
        else
        {
            *numEngines = 0;
            for(auto engineId : allEngineIds)
            {
                if(*numEngines == maxEngines)
                {
                    *numEngines = static_cast<uint32_t>(allEngineIds.size());
                    HIPDNN_PLUGIN_LOG_INFO("Maximum number of engines reached ("
                                           << maxEngines
                                           << "), ignoring additional engines, numEngines count: "
                                           << *numEngines);
                    break;
                }

                engineIds[*numEngines] = engineId;
                (*numEngines)++;
            }
        }

        LOG_API_SUCCESS(apiName, "numEngines=" << *numEngines);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginCreateImpl(hipdnnEnginePluginHandle_t* handle)
{
    LOG_API_ENTRY("handle_ptr=" << static_cast<void*>(handle));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        hipblaslt_plugin::HipblasltHandleFactory::createHipblasltHandle(handle);

        auto hipblasltContainerPtr = hipblasltContainerLifecyclePtr.lock();
        if(hipblasltContainerPtr != nullptr)
        {
            (*handle)->hipblasltContainer = hipblasltContainerPtr;
        }
        else
        {
            static std::mutex s_hipblasltContainerMutex;
            std::lock_guard<std::mutex> lock(s_hipblasltContainerMutex);

            // if we do have a race condition that results in threads getting locked, we want to
            // ensure that we only create one instance.  Therefore, the second thread to get
            // through will just read from the weak pointer rather than create a new instance.
            hipblasltContainerPtr = hipblasltContainerLifecyclePtr.lock();
            if(hipblasltContainerPtr != nullptr)
            {
                (*handle)->hipblasltContainer = hipblasltContainerPtr;
            }
            else
            {
                (*handle)->hipblasltContainer = std::make_shared<HipblasltContainer>();
                hipblasltContainerLifecyclePtr = (*handle)->hipblasltContainer;
            }
        }

        LOG_API_SUCCESS(apiName, "createdHandle=" << static_cast<void*>(*handle));
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyImpl(hipdnnEnginePluginHandle_t handle)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        hipblaslt_plugin::HipblasltHandleFactory::destroyHipblasltHandle(handle);
        delete handle;
        handle = nullptr;

        LOG_API_SUCCESS(apiName, "");
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginSetStreamImpl(hipdnnEnginePluginHandle_t handle,
                                                     hipStream_t stream)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", stream_id=" << static_cast<void*>(stream));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        handle->setStream(stream);

        LOG_API_SUCCESS(apiName, "");
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIdsImpl(hipdnnEnginePluginHandle_t handle,
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
        throwIfNull(handle);
        throwIfNull(opGraph);
        if(maxEngines != 0)
        {
            throwIfNull(engineIds);
        }
        throwIfNull(numEngines);

        auto& engineManager = handle->getEngineManager();
        GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);

        auto applicableEngines = engineManager.getApplicableEngineIds(*handle, opGraphWrapper);

        *numEngines = 0;
        for(auto& engineId : applicableEngines)
        {
            if(*numEngines == maxEngines)
            {
                *numEngines = static_cast<uint32_t>(applicableEngines.size());
                HIPDNN_PLUGIN_LOG_INFO("Maximum number of engines reached ("
                                       << maxEngines
                                       << "), ignoring additional engines, numEngines count: "
                                       << *numEngines);
                break;
            }

            engineIds[*numEngines] = engineId;
            (*numEngines)++;
        }

        LOG_API_SUCCESS(apiName, "numEngines=" << *numEngines);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetEngineDetailsImpl(hipdnnEnginePluginHandle_t handle,
                                                            int64_t engineId,
                                                            const hipdnnPluginConstData_t* opGraph,
                                                            hipdnnPluginConstData_t* engineDetails)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle) << ", engineId=" << engineId
                            << ", opGraph=" << static_cast<const void*>(opGraph)
                            << ", engineDetails=" << static_cast<void*>(engineDetails));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(opGraph);
        throwIfNull(engineDetails);

        auto& engineManager = handle->getEngineManager();
        GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);

        engineManager.getEngineDetails(*handle, opGraphWrapper, engineId, *engineDetails);

        LOG_API_SUCCESS(apiName, "engineDetails->ptr=" << engineDetails->ptr);
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyEngineDetailsImpl(hipdnnEnginePluginHandle_t handle,
                                               hipdnnPluginConstData_t* engineDetails)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", engineDetails=" << static_cast<void*>(engineDetails));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(engineDetails);
        throwIfNull(engineDetails->ptr);

        handle->removeEngineDetailsDetachedBuffer(engineDetails->ptr);

        LOG_API_SUCCESS(apiName, "engineDetails->ptr=" << engineDetails->ptr);
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSizeImpl(hipdnnEnginePluginHandle_t handle,
                                           const hipdnnPluginConstData_t* engineConfig,
                                           const hipdnnPluginConstData_t* opGraph,
                                           size_t* workspaceSize)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", engineConfig=" << static_cast<const void*>(engineConfig)
                            << ", opGraph=" << static_cast<const void*>(opGraph)
                            << ", workspaceSize=" << static_cast<void*>(workspaceSize));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(engineConfig);
        throwIfNull(opGraph);
        throwIfNull(workspaceSize);

        auto& engineManager = handle->getEngineManager();

        EngineConfigWrapper engineConfigWrapper(engineConfig->ptr, engineConfig->size);
        GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);
        *workspaceSize = engineManager.getWorkspaceSize(
            *handle, engineConfigWrapper.engineId(), opGraphWrapper);

        LOG_API_SUCCESS(apiName, "workspaceSize=" << *workspaceSize);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContextImpl(
    hipdnnEnginePluginHandle_t handle,
    const hipdnnPluginConstData_t* engineConfig,
    const hipdnnPluginConstData_t* opGraph,
    hipdnnEnginePluginExecutionContext_t* executionContext)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", engineConfig=" << static_cast<const void*>(engineConfig)
                            << ", opGraph=" << static_cast<const void*>(opGraph)
                            << ", executionContext=" << static_cast<void*>(executionContext));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(engineConfig);
        throwIfNull(opGraph);
        throwIfNull(executionContext);

        GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);
        EngineConfigWrapper engineConfigWrapper(engineConfig->ptr, engineConfig->size);

        auto& engineManager = handle->getEngineManager();

        auto context = new HipdnnEnginePluginExecutionContext;

        try
        {
            engineManager.initializeExecutionContext(
                *handle, opGraphWrapper, engineConfigWrapper, *context);
        }
        catch(...)
        {
            delete context;
            throw;
        }

        *executionContext = context;

        LOG_API_SUCCESS(apiName,
                        "created_execution_context=" << static_cast<void*>(*executionContext));
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContextImpl(
    hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t executionContext)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", executionContext=" << static_cast<void*>(executionContext));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(executionContext);

        delete executionContext;

        LOG_API_SUCCESS(apiName, "destroyed executionContext");
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSizeFromExecutionContextImpl(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext,
    size_t* workspaceSize)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", executionContext=" << static_cast<const void*>(executionContext)
                            << ", workspaceSize=" << static_cast<void*>(workspaceSize));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);
        throwIfNull(executionContext);
        throwIfNull(workspaceSize);

        *workspaceSize = executionContext->plan().getWorkspaceSize(*handle);

        LOG_API_SUCCESS(apiName, "workspaceSize=" << *workspaceSize);
    });
}

hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraphImpl(hipdnnEnginePluginHandle_t handle,
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
        throwIfNull(handle);
        throwIfNull(executionContext);
        throwIfNull(deviceBuffers);

        executionContext->plan().execute(*handle, deviceBuffers, numDeviceBuffers, workspace);

        LOG_API_SUCCESS(apiName, "executed graph");
    });
}

} // extern "C"
