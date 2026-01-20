// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <mutex>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/EnginePluginContainer.hpp>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/PluginHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IExecutionContext.hpp>

/**
 * @file EnginePluginMacros.hpp
 * @brief Macros for implementing engine plugins with minimal boilerplate.
 *
 * This file provides the DECLARE_ENGINE_PLUGIN_DEFAULT_IMPL macro that generates
 * all the C API entry points for an engine plugin. Plugin developers can use this
 * macro to avoid writing boilerplate code.
 *
 * ## Usage
 *
 * 1. Create a container class derived from EnginePluginContainer
 * 2. Create a handle class derived from PluginHandleBase
 * 3. Optionally create an execution context class derived from ExecutionContextBase
 * 4. Use the macro to generate the C API functions
 *
 * ```cpp
 * // MyPlugin.cpp
 * #include <hipdnn_plugin_sdk/EnginePluginMacros.hpp>
 *
 * class MyContainer : public hipdnn_plugin_sdk::EnginePluginContainer {
 *     // Register your engines in the constructor
 * };
 *
 * class MyHandle : public hipdnn_plugin_sdk::PluginHandleBase<MyContainer> {
 *     // Add plugin-specific handle state if needed
 * };
 *
 * DECLARE_ENGINE_PLUGIN_DEFAULT_IMPL(
 *     "my_plugin",           // Plugin name
 *     "1.0.0",              // Plugin version
 *     MyContainer,          // Container type
 *     MyHandle,             // Handle type
 *     hipdnn_plugin_sdk::ExecutionContextBase  // Execution context type
 * )
 * ```
 */

namespace hipdnn_plugin_sdk
{

/**
 * @brief Base class for plugin handles.
 *
 * Plugin developers should derive from this class to create their handle type.
 * The base class provides container access and stream management.
 *
 * @tparam ContainerType The container class derived from EnginePluginContainer.
 */
template<typename ContainerType>
struct PluginHandleBase
{
    virtual ~PluginHandleBase() = default;

    /// The shared container instance
    std::shared_ptr<ContainerType> container;

    /// The HIP stream for this handle
    hipStream_t stream = nullptr;

    /**
     * @brief Sets the HIP stream for this handle.
     * @param newStream The stream to set.
     */
    virtual void setStream(hipStream_t newStream)
    {
        stream = newStream;
    }

    /**
     * @brief Gets the current HIP stream.
     * @return The current stream.
     */
    hipStream_t getStream() const
    {
        return stream;
    }

    /**
     * @brief Gets the engine manager from the container.
     * @return Reference to the engine manager.
     */
    EngineManager& getEngineManager()
    {
        return container->getEngineManager();
    }
};

} // namespace hipdnn_plugin_sdk

// NOLINTBEGIN(cppcoreguidelines-macro-usage)

/**
 * @brief Declares default implementations for all engine plugin C API functions.
 *
 * This macro generates the complete implementation of all required C API entry
 * points for an engine plugin. It handles:
 * - Plugin metadata (name, version, type)
 * - Error handling and logging
 * - Handle creation/destruction with shared container lifecycle
 * - Execution context management
 * - Graph execution delegation to the engine manager
 *
 * @param PLUGIN_NAME String literal for the plugin name
 * @param PLUGIN_VERSION String literal for the plugin version
 * @param CONTAINER_TYPE The container class (must derive from EnginePluginContainer)
 * @param HANDLE_TYPE The handle class (must derive from PluginHandleBase<CONTAINER_TYPE>)
 * @param CONTEXT_TYPE The execution context class (must derive from IExecutionContext)
 */
#define DECLARE_ENGINE_PLUGIN_DEFAULT_IMPL(                                                        \
    PLUGIN_NAME, PLUGIN_VERSION, CONTAINER_TYPE, HANDLE_TYPE, CONTEXT_TYPE)                        \
                                                                                                   \
    /* Static plugin metadata */                                                                   \
    static const char* g_pluginName = PLUGIN_NAME;                                                 \
    static const char* g_pluginVersion = PLUGIN_VERSION;                                           \
                                                                                                   \
    /* Thread-local error string storage */                                                        \
    thread_local char                                                                              \
        hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = ""; \
                                                                                                   \
    /* Shared container manager */                                                                 \
    static hipdnn_plugin_sdk::SharedContainerManager<CONTAINER_TYPE> g_containerManager;           \
                                                                                                   \
    extern "C" {                                                                                   \
                                                                                                   \
    /* Base plugin API functions */                                                                \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)                                    \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(name);                                                  \
            *name = g_pluginName;                                                                  \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)                              \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(version);                                               \
            *version = g_pluginVersion;                                                            \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)                             \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(type);                                                  \
            *type = HIPDNN_PLUGIN_TYPE_ENGINE;                                                     \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    void hipdnnPluginGetLastErrorString(const char** errorStr)                                     \
    {                                                                                              \
        hipdnn_plugin_sdk::tryCatch([&]() {                                                        \
            hipdnn_plugin_sdk::throwIfNull(errorStr);                                              \
            *errorStr = hipdnn_plugin_sdk::PluginLastErrorManager::getLastError();                 \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)                 \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(callback);                                              \
            hipdnn::logging::initializeCallbackLogging(g_pluginName, callback);                    \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    /* Engine plugin API functions */                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginGetAllEngineIds(                                        \
        int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)                             \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            if(maxEngines != 0)                                                                    \
            {                                                                                      \
                hipdnn_plugin_sdk::throwIfNull(engineIds);                                         \
            }                                                                                      \
            hipdnn_plugin_sdk::throwIfNull(numEngines);                                            \
                                                                                                   \
            auto container = g_containerManager.getOrCreate();                                     \
            auto allEngineIds = container->getEngineManager().getAllEngineIds();                   \
                                                                                                   \
            *numEngines = 0;                                                                       \
            for(auto engineId : allEngineIds)                                                      \
            {                                                                                      \
                if(*numEngines >= maxEngines)                                                      \
                {                                                                                  \
                    *numEngines = static_cast<uint32_t>(allEngineIds.size());                      \
                    break;                                                                         \
                }                                                                                  \
                engineIds[*numEngines] = engineId;                                                 \
                (*numEngines)++;                                                                   \
            }                                                                                      \
            if(*numEngines < allEngineIds.size())                                                  \
            {                                                                                      \
                *numEngines = static_cast<uint32_t>(allEngineIds.size());                          \
            }                                                                                      \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)              \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(handle);                                                \
            auto* newHandle = new HANDLE_TYPE();                                                   \
            newHandle->container = g_containerManager.getOrCreate();                               \
            *handle = newHandle;                                                                   \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)              \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(handle);                                                \
            delete handle;                                                                         \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginSetStream(                                              \
        hipdnnEnginePluginHandle_t handle, hipStream_t stream)                                     \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(handle);                                                \
            handle->setStream(stream);                                                             \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginGetApplicableEngineIds(                                 \
        hipdnnEnginePluginHandle_t handle,                                                         \
        const hipdnnPluginConstData_t* opGraph,                                                    \
        int64_t* engineIds,                                                                        \
        uint32_t maxEngines,                                                                       \
        uint32_t* numEngines)                                                                      \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(handle);                                                \
            hipdnn_plugin_sdk::throwIfNull(opGraph);                                               \
            if(maxEngines != 0)                                                                    \
            {                                                                                      \
                hipdnn_plugin_sdk::throwIfNull(engineIds);                                         \
            }                                                                                      \
            hipdnn_plugin_sdk::throwIfNull(numEngines);                                            \
                                                                                                   \
            hipdnn_plugin_sdk::GraphWrapper graphWrapper(opGraph->ptr, opGraph->size);             \
            auto applicable = handle->getEngineManager().getApplicableEngineIds(handle, graphWrapper); \
                                                                                                   \
            *numEngines = 0;                                                                       \
            for(auto engineId : applicable)                                                        \
            {                                                                                      \
                if(*numEngines >= maxEngines)                                                      \
                {                                                                                  \
                    *numEngines = static_cast<uint32_t>(applicable.size());                        \
                    break;                                                                         \
                }                                                                                  \
                engineIds[*numEngines] = engineId;                                                 \
                (*numEngines)++;                                                                   \
            }                                                                                      \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginGetEngineDetails(                                       \
        hipdnnEnginePluginHandle_t handle,                                                         \
        int64_t engineId,                                                                          \
        const hipdnnPluginConstData_t* opGraph,                                                    \
        hipdnnPluginConstData_t* engineDetails)                                                    \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(handle);                                                \
            hipdnn_plugin_sdk::throwIfNull(opGraph);                                               \
            hipdnn_plugin_sdk::throwIfNull(engineDetails);                                         \
                                                                                                   \
            hipdnn_plugin_sdk::GraphWrapper graphWrapper(opGraph->ptr, opGraph->size);             \
            handle->getEngineManager().getEngineDetails(handle, graphWrapper, engineId, *engineDetails); \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginDestroyEngineDetails(                                   \
        hipdnnEnginePluginHandle_t handle, hipdnnPluginConstData_t* engineDetails)                 \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(handle);                                                \
            hipdnn_plugin_sdk::throwIfNull(engineDetails);                                         \
            /* Default implementation: engine details memory is managed by the handle */          \
            /* Plugin-specific handles can override this behavior */                               \
            engineDetails->ptr = nullptr;                                                          \
            engineDetails->size = 0;                                                               \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSize(                                       \
        hipdnnEnginePluginHandle_t handle,                                                         \
        const hipdnnPluginConstData_t* engineConfig,                                               \
        const hipdnnPluginConstData_t* opGraph,                                                    \
        size_t* workspaceSize)                                                                     \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(handle);                                                \
            hipdnn_plugin_sdk::throwIfNull(engineConfig);                                          \
            hipdnn_plugin_sdk::throwIfNull(opGraph);                                               \
            hipdnn_plugin_sdk::throwIfNull(workspaceSize);                                         \
                                                                                                   \
            hipdnn_plugin_sdk::EngineConfigWrapper configWrapper(engineConfig->ptr, engineConfig->size); \
            hipdnn_plugin_sdk::GraphWrapper graphWrapper(opGraph->ptr, opGraph->size);             \
            *workspaceSize = handle->getEngineManager().getWorkspaceSize(                          \
                handle, configWrapper.engineId(), graphWrapper);                                   \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContext(                                 \
        hipdnnEnginePluginHandle_t handle,                                                         \
        const hipdnnPluginConstData_t* engineConfig,                                               \
        const hipdnnPluginConstData_t* opGraph,                                                    \
        hipdnnEnginePluginExecutionContext_t* executionContext)                                    \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(handle);                                                \
            hipdnn_plugin_sdk::throwIfNull(engineConfig);                                          \
            hipdnn_plugin_sdk::throwIfNull(opGraph);                                               \
            hipdnn_plugin_sdk::throwIfNull(executionContext);                                      \
                                                                                                   \
            hipdnn_plugin_sdk::EngineConfigWrapper configWrapper(engineConfig->ptr, engineConfig->size); \
            hipdnn_plugin_sdk::GraphWrapper graphWrapper(opGraph->ptr, opGraph->size);             \
                                                                                                   \
            auto* context = new CONTEXT_TYPE();                                                    \
            try                                                                                    \
            {                                                                                      \
                handle->getEngineManager().initializeExecutionContext(                             \
                    handle, graphWrapper, configWrapper, *context);                                \
            }                                                                                      \
            catch(...)                                                                             \
            {                                                                                      \
                delete context;                                                                    \
                throw;                                                                             \
            }                                                                                      \
            *executionContext = context;                                                           \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(                                \
        hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t executionContext)  \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(handle);                                                \
            hipdnn_plugin_sdk::throwIfNull(executionContext);                                      \
            delete executionContext;                                                               \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(                   \
        hipdnnEnginePluginHandle_t handle,                                                         \
        hipdnnEnginePluginExecutionContext_t executionContext,                                     \
        size_t* workspaceSize)                                                                     \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(handle);                                                \
            hipdnn_plugin_sdk::throwIfNull(executionContext);                                      \
            hipdnn_plugin_sdk::throwIfNull(workspaceSize);                                         \
                                                                                                   \
            *workspaceSize = executionContext->getPlan().getWorkspaceSize(handle);                 \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    hipdnnPluginStatus_t hipdnnEnginePluginExecuteOpGraph(                                         \
        hipdnnEnginePluginHandle_t handle,                                                         \
        hipdnnEnginePluginExecutionContext_t executionContext,                                     \
        void* workspace,                                                                           \
        const hipdnnPluginDeviceBuffer_t* deviceBuffers,                                           \
        uint32_t numDeviceBuffers)                                                                 \
    {                                                                                              \
        return hipdnn_plugin_sdk::tryCatch([&]() {                                                 \
            hipdnn_plugin_sdk::throwIfNull(handle);                                                \
            hipdnn_plugin_sdk::throwIfNull(executionContext);                                      \
            hipdnn_plugin_sdk::throwIfNull(deviceBuffers);                                         \
                                                                                                   \
            executionContext->getPlan().execute(handle, deviceBuffers, numDeviceBuffers, workspace); \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    } /* extern "C" */

// NOLINTEND(cppcoreguidelines-macro-usage)
