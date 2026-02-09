// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/PluginDataTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>

#include "EngineManager.hpp"
#include "HipKernelContainer.hpp"
#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"

static const char* pluginName = "hip_kernel_plugin";
static const char* pluginVersion = "1.0.0";

using namespace hipdnn_data_sdk::flatbuffer_utilities;
using namespace hipdnn_plugin_sdk;
using namespace hip_kernel_plugin;

// NOLINTNEXTLINE
thread_local char PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

// Keep a weak pointer to the HipKernelContainer thats made when we create a plugin handle.
// The original shared_ptr is then stored on the handle so that it can be used for the lifecycle
// of the handle.  If we create another handle, then we can use the weak pointer to get access
// to the existing HipKernelContainer.  If all handles are destroyed, then this allows us to properly
// clean up the container without having to fully unload the plugin.
std::weak_ptr<HipKernelContainer> hipKernelContainerLifecyclePtr;

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetNameImpl(const char** name)
{
    LOG_API_ENTRY("name_ptr=" << static_cast<const void*>(name));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(name);

        *name = pluginName;

        LOG_API_SUCCESS(apiName, "pluginName=" << static_cast<const void*>(name));
    });
}

hipdnnPluginStatus_t hipdnnPluginGetVersionImpl(const char** version)
{
    LOG_API_ENTRY("versionPtr=" << static_cast<const void*>(version));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(version);

        *version = pluginVersion;

        LOG_API_SUCCESS(apiName, "version=" << static_cast<const void*>(version));
    });
}

hipdnnPluginStatus_t hipdnnPluginGetTypeImpl(hipdnnPluginType_t* type)
{
    LOG_API_ENTRY("typePtr=" << static_cast<void*>(type));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(type);

        *type = HIPDNN_PLUGIN_TYPE_ENGINE;

        LOG_API_SUCCESS(apiName, "type=" << *type);
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
        hipdnn::logging::initializeCallbackLogging(pluginName, callback);
        LOG_API_SUCCESS(apiName, "");
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

        auto totalEngines = HipKernelContainer::copyEngineIds(engineIds, maxEngines, *numEngines);

        LOG_API_SUCCESS(apiName, "numEngines=" << *numEngines << " totalEngines=" << totalEngines);
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginCreateImpl(hipdnnEnginePluginHandle_t* handle)
{
    LOG_API_ENTRY("handle_ptr=" << static_cast<void*>(handle));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(handle);

        *handle = new HipdnnEnginePluginHandle();

        auto hipKernelContainerPtr = hipKernelContainerLifecyclePtr.lock();
        if(hipKernelContainerPtr != nullptr)
        {
            (*handle)->hipKernelContainer = hipKernelContainerPtr;
        }
        else
        {
            static std::mutex s_hipKernelContainerMutex;
            std::lock_guard<std::mutex> lock(s_hipKernelContainerMutex);

            // if we do have a race condition that results in threads getting locked, we want to
            // ensure that we only create one instance.  Therefore, the second thread to get
            // through will just read from the weak pointer rather than create a new instance.
            hipKernelContainerPtr = hipKernelContainerLifecyclePtr.lock();
            if(hipKernelContainerPtr != nullptr)
            {
                (*handle)->hipKernelContainer = hipKernelContainerPtr;
            }
            else
            {
                (*handle)->hipKernelContainer = std::make_shared<HipKernelContainer>();
                hipKernelContainerLifecyclePtr = (*handle)->hipKernelContainer;
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
