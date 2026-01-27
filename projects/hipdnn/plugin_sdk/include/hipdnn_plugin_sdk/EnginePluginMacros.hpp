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
 * 1. Create a container class with the following requirements:
 *    - Must have: EngineManager& getEngineManager()
 *    - Must have: static uint32_t copyEngineIds(int64_t*, uint32_t, uint32_t&)
 *
 * 2. Define HipdnnEnginePluginHandle struct with:
 *    - void setStream(hipStream_t)
 *    - EngineManager& getEngineManager()
 *    - std::shared_ptr<ContainerType> to hold container instance
 *
 * 3. Define HipdnnEnginePluginExecutionContext struct with:
 *    - IPlan& plan()
 *
 * 4. Use the macro to generate the C API functions
 *
 * ```cpp
 * // MyPlugin.cpp
 * #include <hipdnn_plugin_sdk/EnginePluginMacros.hpp>
 *
 * class MyContainer {
 * public:
 *     static uint32_t copyEngineIds(int64_t*, uint32_t, uint32_t&);
 *     EngineManager& getEngineManager();
 * };
 *
 * struct HipdnnEnginePluginHandle {
 *     void setStream(hipStream_t stream);
 *     EngineManager& getEngineManager();
 *     std::shared_ptr<MyContainer> container;
 * };
 *
 * struct HipdnnEnginePluginExecutionContext {
 *     IPlan& plan();
 * };
 *
 * DECLARE_ENGINE_PLUGIN_DEFAULT_IMPL(
 *     "my_plugin",           // Plugin name
 *     "1.0.0",              // Plugin version
 *     MyContainer           // Container type
 * )
 * ```
 */

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
 * The macro expects HipdnnEnginePluginHandle and HipdnnEnginePluginExecutionContext
 * to be defined by the plugin.
 *
 * @param PLUGIN_NAME String literal for the plugin name
 * @param PLUGIN_VERSION String literal for the plugin version
 * @param CONTAINER_TYPE The container class (must have getEngineManager() and static copyEngineIds())
 */
#define DECLARE_ENGINE_PLUGIN_DEFAULT_IMPL(PLUGIN_NAME, PLUGIN_VERSION, CONTAINER_TYPE)          \
                                                                                                  \
    /* Compile-time validation of container type */                                              \
    namespace                                                                                     \
    {                                                                                             \
    [[maybe_unused]] constexpr bool container_validation                                          \
        = (hipdnn_plugin_sdk::validateContainerType<CONTAINER_TYPE>(), true);                     \
    }                                                                                             \
                                                                                                  \
    /* Static plugin metadata */                                                                  \
    static const char* g_pluginName = PLUGIN_NAME;                                                \
    static const char* g_pluginVersion = PLUGIN_VERSION;                                          \
                                                                                                  \
    /* Thread-local error string storage */                                                       \
    /* NOLINTNEXTLINE(modernize-avoid-c-arrays) */                                                \
    thread_local char hipdnn_plugin_sdk::PluginLastErrorManager::s_lastError                      \
        [HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH]                                                   \
        = "";                                                                                     \
                                                                                                  \
    /* Shared container manager for container lifecycle */                                        \
    static hipdnn_plugin_sdk::SharedContainerManager<CONTAINER_TYPE> g_containerManager;          \
                                                                                                  \
    extern "C" {                                                                                  \
                                                                                                  \
    /* Base plugin API functions */                                                               \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)                                   \
    {                                                                                             \
        LOG_API_ENTRY("name_ptr={:p}", static_cast<void*>(name));                                 \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(name);                                                 \
                                                                                                  \
            *name = g_pluginName;                                                                 \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "pluginName={:p}", static_cast<void*>(name));                \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)                             \
    {                                                                                             \
        LOG_API_ENTRY("versionPtr={:p}", static_cast<void*>(version));                            \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(version);                                              \
                                                                                                  \
            *version = g_pluginVersion;                                                           \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "version={:p}", static_cast<void*>(version));                \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)                            \
    {                                                                                             \
        LOG_API_ENTRY("typePtr={:p}", static_cast<void*>(type));                                  \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(type);                                                 \
                                                                                                  \
            *type = HIPDNN_PLUGIN_TYPE_ENGINE;                                                    \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "type={}", *type);                                           \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    void hipdnnPluginGetLastErrorString(const char** errorStr)                                    \
    {                                                                                             \
        LOG_API_ENTRY("errorStrPtr={:p}", static_cast<void*>(errorStr));                          \
                                                                                                  \
        hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                                   \
            hipdnn_plugin_sdk::throwIfNull(errorStr);                                             \
                                                                                                  \
            *errorStr = hipdnn_plugin_sdk::PluginLastErrorManager::getLastError();                \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "errorStr={:p}", static_cast<void*>(errorStr));              \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)                \
    {                                                                                             \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(callback);                                             \
            hipdnn::logging::initializeCallbackLogging(g_pluginName, callback);                   \
            LOG_API_SUCCESS(apiName, "", "");                                                     \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    /* Engine plugin API functions */                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds,                    \
                                                           uint32_t maxEngines,                   \
                                                           uint32_t* numEngines)                  \
    {                                                                                             \
        LOG_API_ENTRY("engineIds={:p}, maxEngines={}, numEngines={:p}",                          \
                      static_cast<void*>(engineIds),                                              \
                      maxEngines,                                                                 \
                      static_cast<void*>(numEngines));                                            \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            if(maxEngines != 0)                                                                   \
            {                                                                                     \
                hipdnn_plugin_sdk::throwIfNull(engineIds);                                        \
            }                                                                                     \
            hipdnn_plugin_sdk::throwIfNull(numEngines);                                           \
                                                                                                  \
            auto totalEngines = CONTAINER_TYPE::copyEngineIds(engineIds, maxEngines, *numEngines); \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "numEngines={} totalEngines={}", *numEngines, totalEngines); \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)             \
    {                                                                                             \
        LOG_API_ENTRY("handle_ptr={:p}", static_cast<void*>(handle));                             \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(handle);                                               \
                                                                                                  \
            auto* newHandle = new HipdnnEnginePluginHandle();                                     \
            newHandle->container = g_containerManager.getOrCreate();                              \
            *handle = newHandle;                                                                  \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "createdHandle={:p}", static_cast<void*>(*handle));          \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)             \
    {                                                                                             \
        LOG_API_ENTRY("handle={:p}", static_cast<void*>(handle));                                 \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(handle);                                               \
                                                                                                  \
            delete handle;                                                                        \
            handle = nullptr;                                                                     \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "", "");                                                     \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,           \
                                                     hipStream_t stream)                          \
    {                                                                                             \
        LOG_API_ENTRY(                                                                            \
            "handle={:p}, stream_id={:p}", static_cast<void*>(handle), static_cast<void*>(stream)); \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(handle);                                               \
                                                                                                  \
            handle->setStream(stream);                                                            \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "", "");                                                     \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,               \
                                                 const hipdnnPluginConstData_t* opGraph,          \
                                                 int64_t* engineIds,                              \
                                                 uint32_t maxEngines,                             \
                                                 uint32_t* numEngines)                            \
    {                                                                                             \
        LOG_API_ENTRY("handle={:p}, opGraph={:p}, engineIds={:p}, maxEngines={}, numEngines={:p}", \
                      static_cast<void*>(handle),                                                 \
                      static_cast<const void*>(opGraph),                                          \
                      static_cast<void*>(engineIds),                                              \
                      maxEngines,                                                                 \
                      static_cast<void*>(numEngines));                                            \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(handle);                                               \
            hipdnn_plugin_sdk::throwIfNull(opGraph);                                              \
            if(maxEngines != 0)                                                                   \
            {                                                                                     \
                hipdnn_plugin_sdk::throwIfNull(engineIds);                                        \
            }                                                                                     \
            hipdnn_plugin_sdk::throwIfNull(numEngines);                                           \
                                                                                                  \
            auto& engineManager = handle->getEngineManager();                                     \
            hipdnn_plugin_sdk::GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);          \
                                                                                                  \
            auto applicableEngines = engineManager.getApplicableEngineIds(*handle, opGraphWrapper); \
                                                                                                  \
            *numEngines = 0;                                                                      \
            for(auto& engineId : applicableEngines)                                               \
            {                                                                                     \
                if(*numEngines == maxEngines)                                                     \
                {                                                                                 \
                    *numEngines = static_cast<uint32_t>(applicableEngines.size());                \
                    HIPDNN_LOG_INFO("Maximum number of engines reached ({}), ignoring additional " \
                                    "engines, numEngines count: {}",                              \
                                    maxEngines,                                                   \
                                    *numEngines);                                                 \
                    break;                                                                        \
                }                                                                                 \
                                                                                                  \
                engineIds[*numEngines] = engineId;                                                \
                (*numEngines)++;                                                                  \
            }                                                                                     \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "numEngines={}", *numEngines);                               \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,                     \
                                           int64_t engineId,                                      \
                                           const hipdnnPluginConstData_t* opGraph,                \
                                           hipdnnPluginConstData_t* engineDetails)                \
    {                                                                                             \
        LOG_API_ENTRY("handle={:p}, engineId={}, opGraph={:p}, engineDetails={:p}",              \
                      static_cast<void*>(handle),                                                 \
                      engineId,                                                                   \
                      static_cast<const void*>(opGraph),                                          \
                      static_cast<void*>(engineDetails));                                         \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(handle);                                               \
            hipdnn_plugin_sdk::throwIfNull(opGraph);                                              \
            hipdnn_plugin_sdk::throwIfNull(engineDetails);                                        \
                                                                                                  \
            auto& engineManager = handle->getEngineManager();                                     \
            hipdnn_plugin_sdk::GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);          \
                                                                                                  \
            engineManager.getEngineDetails(*handle, opGraphWrapper, engineId, *engineDetails);    \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "engineDetails->ptr={:p}", engineDetails->ptr);              \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,                 \
                                               hipdnnPluginConstData_t* engineDetails)            \
    {                                                                                             \
        LOG_API_ENTRY("handle={:p}, engineDetails={}",                                            \
                      static_cast<void*>(handle),                                                 \
                      static_cast<void*>(engineDetails));                                         \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(handle);                                               \
            hipdnn_plugin_sdk::throwIfNull(engineDetails);                                        \
            hipdnn_plugin_sdk::throwIfNull(engineDetails->ptr);                                   \
                                                                                                  \
            handle->removeEngineDetailsDetachedBuffer(engineDetails->ptr);                        \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "engineDetails->ptr={:p}", engineDetails->ptr);              \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,                     \
                                           const hipdnnPluginConstData_t* engineConfig,           \
                                           const hipdnnPluginConstData_t* opGraph,                \
                                           size_t* workspaceSize)                                 \
    {                                                                                             \
        LOG_API_ENTRY("handle={:p}, engineConfig={:p}, opGraph={:p}, workspaceSize={:p}",        \
                      static_cast<void*>(handle),                                                 \
                      static_cast<const void*>(engineConfig),                                     \
                      static_cast<const void*>(opGraph),                                          \
                      static_cast<void*>(workspaceSize));                                         \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(handle);                                               \
            hipdnn_plugin_sdk::throwIfNull(engineConfig);                                         \
            hipdnn_plugin_sdk::throwIfNull(opGraph);                                              \
            hipdnn_plugin_sdk::throwIfNull(workspaceSize);                                        \
                                                                                                  \
            auto& engineManager = handle->getEngineManager();                                     \
                                                                                                  \
            hipdnn_plugin_sdk::EngineConfigWrapper engineConfigWrapper(engineConfig->ptr,         \
                                                                       engineConfig->size);       \
            hipdnn_plugin_sdk::GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);          \
            *workspaceSize = engineManager.getWorkspaceSize(                                      \
                *handle, engineConfigWrapper.engineId(), opGraphWrapper);                         \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "workspaceSize={}", *workspaceSize);                         \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContext(                                \
        hipdnnEnginePluginHandle_t handle,                                                        \
        const hipdnnPluginConstData_t* engineConfig,                                              \
        const hipdnnPluginConstData_t* opGraph,                                                   \
        hipdnnEnginePluginExecutionContext_t* executionContext)                                   \
    {                                                                                             \
        LOG_API_ENTRY("handle={:p}, engineConfig={:p}, opGraph={:p}, executionContext={:p}",     \
                      static_cast<void*>(handle),                                                 \
                      static_cast<const void*>(engineConfig),                                     \
                      static_cast<const void*>(opGraph),                                          \
                      static_cast<void*>(executionContext));                                      \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(handle);                                               \
            hipdnn_plugin_sdk::throwIfNull(engineConfig);                                         \
            hipdnn_plugin_sdk::throwIfNull(opGraph);                                              \
            hipdnn_plugin_sdk::throwIfNull(executionContext);                                     \
                                                                                                  \
            hipdnn_plugin_sdk::GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);          \
            hipdnn_plugin_sdk::EngineConfigWrapper engineConfigWrapper(engineConfig->ptr,         \
                                                                       engineConfig->size);       \
                                                                                                  \
            auto& engineManager = handle->getEngineManager();                                     \
                                                                                                  \
            auto context = new HipdnnEnginePluginExecutionContext;                                \
                                                                                                  \
            try                                                                                   \
            {                                                                                     \
                engineManager.initializeExecutionContext(                                         \
                    *handle, opGraphWrapper, engineConfigWrapper, *context);                      \
            }                                                                                     \
            catch(...)                                                                            \
            {                                                                                     \
                delete context;                                                                   \
                throw;                                                                            \
            }                                                                                     \
                                                                                                  \
            *executionContext = context;                                                          \
                                                                                                  \
            LOG_API_SUCCESS(                                                                      \
                apiName, "created_execution_context={:p}", static_cast<void*>(*executionContext)); \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(                               \
        hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t executionContext) \
    {                                                                                             \
        LOG_API_ENTRY("handle={:p}, executionContext={:p}",                                       \
                      static_cast<void*>(handle),                                                 \
                      static_cast<void*>(executionContext));                                      \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(handle);                                               \
            hipdnn_plugin_sdk::throwIfNull(executionContext);                                     \
                                                                                                  \
            delete executionContext;                                                              \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "destroyed executionContext", "");                           \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(                  \
        hipdnnEnginePluginHandle_t handle,                                                        \
        hipdnnEnginePluginExecutionContext_t executionContext,                                    \
        size_t* workspaceSize)                                                                    \
    {                                                                                             \
        LOG_API_ENTRY("handle={:p}, executionContext={:p}, workspaceSize={:p}",                  \
                      static_cast<void*>(handle),                                                 \
                      static_cast<const void*>(executionContext),                                 \
                      static_cast<void*>(workspaceSize));                                         \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(handle);                                               \
            hipdnn_plugin_sdk::throwIfNull(executionContext);                                     \
            hipdnn_plugin_sdk::throwIfNull(workspaceSize);                                        \
                                                                                                  \
            *workspaceSize = executionContext->plan().getWorkspaceSize(*handle);                  \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "workspaceSize={}", *workspaceSize);                         \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,                       \
                                         hipdnnEnginePluginExecutionContext_t executionContext,   \
                                         void* workspace,                                         \
                                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,         \
                                         uint32_t numDeviceBuffers)                               \
    {                                                                                             \
        LOG_API_ENTRY("handle={:p}, executionContext={:p}, workspace={:p}, deviceBuffers={:p}, " \
                      "numDeviceBuffers={}",                                                      \
                      static_cast<void*>(handle),                                                 \
                      static_cast<void*>(executionContext),                                       \
                      workspace,                                                                  \
                      static_cast<const void*>(deviceBuffers),                                    \
                      numDeviceBuffers);                                                          \
                                                                                                  \
        return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {                            \
            hipdnn_plugin_sdk::throwIfNull(handle);                                               \
            hipdnn_plugin_sdk::throwIfNull(executionContext);                                     \
            hipdnn_plugin_sdk::throwIfNull(deviceBuffers);                                        \
                                                                                                  \
            executionContext->plan().execute(*handle, deviceBuffers, numDeviceBuffers, workspace); \
                                                                                                  \
            LOG_API_SUCCESS(apiName, "executed graph", "");                                       \
        });                                                                                       \
    }                                                                                             \
                                                                                                  \
    } /* extern "C" */

// NOLINTEND(cppcoreguidelines-macro-usage)
