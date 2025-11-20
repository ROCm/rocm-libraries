// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstring>
#include <iostream>
#include <memory>

#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApi.h>
#include <hipdnn_sdk/plugin/PluginDataTypeHelpers.hpp>
#include <hipdnn_sdk/plugin/PluginHelpers.hpp>
#include <hipdnn_sdk/plugin/PluginLastErrorManager.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>

struct HipdnnEnginePluginHandle
{
public:
    virtual ~HipdnnEnginePluginHandle() = default;
};

struct HipdnnEnginePluginExecutionContext
{
};

// Base class for test plugins
class TestPluginBase
{
public:
    virtual ~TestPluginBase() = default;

    // Virtual methods to be overridden by derived classes
    virtual const char* getPluginName() const = 0;
    virtual const char* getPluginVersion() const = 0;
    virtual int64_t getEngineId() const = 0;
    virtual uint32_t getNumEngines() const = 0;
    virtual uint32_t getNumApplicableEngines() const = 0;
    virtual bool supportsEngineOperations() const
    {
        return getNumApplicableEngines() > 0;
    }

    // Execute graph - derived classes override this for custom behavior
    virtual void executeGraph() const
    {
        HIPDNN_LOG_INFO("executeGraph called");
    }

    // Static instance management
    static void setInstance(std::unique_ptr<TestPluginBase> instance)
    {
        s_instance = std::move(instance);
    }

    static TestPluginBase* getInstance()
    {
        return s_instance.get();
    }

    // Common API implementations
    static hipdnnPluginStatus_t pluginGetName(const char** name)
    {
        LOG_API_ENTRY("namePtr={:p}", static_cast<void*>(name));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(name);
            hipdnn_plugin::throwIfNull(getInstance());

            *name = getInstance()->getPluginName();

            LOG_API_SUCCESS(apiName, "pluginName={:p}", static_cast<void*>(name));
        });
    }

    static hipdnnPluginStatus_t pluginGetVersion(const char** version)
    {
        LOG_API_ENTRY("versionPtr={:p}", static_cast<void*>(version));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(version);
            hipdnn_plugin::throwIfNull(getInstance());

            *version = getInstance()->getPluginVersion();

            LOG_API_SUCCESS(apiName, "version={:p}", static_cast<void*>(version));
        });
    }

    static hipdnnPluginStatus_t pluginGetType(hipdnnPluginType_t* type)
    {
        LOG_API_ENTRY("typePtr={:p}", static_cast<void*>(type));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(type);

            *type = HIPDNN_PLUGIN_TYPE_ENGINE;

            LOG_API_SUCCESS(apiName, "type={}", *type);
        });
    }

    static void pluginGetLastErrorString(const char** errorStr)
    {
        LOG_API_ENTRY("errorStrPtr={:p}", static_cast<void*>(errorStr));

        hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(errorStr);

            *errorStr = hipdnn_plugin::PluginLastErrorManager::getLastError();

            LOG_API_SUCCESS(apiName, "errorStr={:p}", static_cast<void*>(errorStr));
        });
    }

    static hipdnnPluginStatus_t pluginSetLoggingCallback(hipdnnCallback_t callback)
    {
        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(callback);
            hipdnn_plugin::throwIfNull(getInstance());

            hipdnn::logging::initializeCallbackLogging(COMPONENT_NAME, callback);
            LOG_API_SUCCESS(apiName, "", "");
        });
    }

    static hipdnnPluginStatus_t
        enginePluginGetAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
    {
        LOG_API_ENTRY("engineIds={:p}, maxEngines={}, numEngines={:p}",
                      static_cast<void*>(engineIds),
                      maxEngines,
                      static_cast<void*>(numEngines));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            if(maxEngines != 0)
            {
                hipdnn_plugin::throwIfNull(engineIds);
            }
            hipdnn_plugin::throwIfNull(numEngines);
            hipdnn_plugin::throwIfNull(getInstance());

            *numEngines = getInstance()->getNumEngines();

            if(maxEngines >= 1 && *numEngines > 0)
            {
                assert(*numEngines == 1);
                engineIds[0] = getInstance()->getEngineId();
            }

            LOG_API_SUCCESS(apiName, "numEngines={}", *numEngines);
        });
    }

    static hipdnnPluginStatus_t enginePluginCreate(hipdnnEnginePluginHandle_t* handle)
    {
        LOG_API_ENTRY("handlePtr={:p}", static_cast<void*>(handle));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(handle);

            *handle = new HipdnnEnginePluginHandle();

            LOG_API_SUCCESS(apiName, "createdHandle={:p}", static_cast<void*>(*handle));
        });
    }

    static hipdnnPluginStatus_t enginePluginDestroy(hipdnnEnginePluginHandle_t handle)
    {
        LOG_API_ENTRY("handle={:p}", static_cast<void*>(handle));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(handle);

            delete handle;
            handle = nullptr;

            LOG_API_SUCCESS(apiName, "", "");
        });
    }

    static hipdnnPluginStatus_t enginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                                                      hipStream_t stream)
    {
        LOG_API_ENTRY(
            "handle={:p}, streamId={:p}", static_cast<void*>(handle), static_cast<void*>(stream));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(handle);

            LOG_API_SUCCESS(apiName, "", "");
        });
    }

    static hipdnnPluginStatus_t
        enginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                           const hipdnnPluginConstData_t* opGraph,
                                           int64_t* engineIds,
                                           uint32_t maxEngines,
                                           uint32_t* numEngines)
    {
        LOG_API_ENTRY("handle={:p}, opGraph={:p}, engineIds={:p}, maxEngines={}, numEngines={:p}",
                      static_cast<void*>(handle),
                      static_cast<const void*>(opGraph),
                      static_cast<void*>(engineIds),
                      maxEngines,
                      static_cast<void*>(numEngines));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(handle);
            hipdnn_plugin::throwIfNull(opGraph);
            if(maxEngines != 0)
            {
                hipdnn_plugin::throwIfNull(engineIds);
            }
            hipdnn_plugin::throwIfNull(numEngines);
            hipdnn_plugin::throwIfNull(getInstance());

            *numEngines = getInstance()->getNumApplicableEngines();

            if(maxEngines >= 1 && *numEngines > 0)
            {
                engineIds[0] = getInstance()->getEngineId();
            }

            LOG_API_SUCCESS(apiName, "numEngines={}", *numEngines);
        });
    }

    static hipdnnPluginStatus_t enginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                             int64_t engineId,
                                                             const hipdnnPluginConstData_t* opGraph,
                                                             hipdnnPluginConstData_t* engineDetails)
    {
        LOG_API_ENTRY("handle={:p}, engineId={}, opGraph={:p}, engineDetails={:p}",
                      static_cast<void*>(handle),
                      engineId,
                      static_cast<const void*>(opGraph),
                      static_cast<void*>(engineDetails));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(handle);
            hipdnn_plugin::throwIfNull(opGraph);
            hipdnn_plugin::throwIfNull(engineDetails);
            hipdnn_plugin::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot get engine details");
            }

            flatbuffers::FlatBufferBuilder builder;
            auto newEngineDetails = hipdnn_sdk::data_objects::CreateEngineDetails(
                builder, getInstance()->getEngineId());
            builder.Finish(newEngineDetails);
            auto serializedDetails = builder.Release();

            auto* tempBuffer = new uint8_t[serializedDetails.size()];
            std::memcpy(tempBuffer, serializedDetails.data(), serializedDetails.size());

            engineDetails->ptr = tempBuffer;
            engineDetails->size = serializedDetails.size();

            LOG_API_SUCCESS(apiName, "engineDetails->ptr={:p}", engineDetails->ptr);
        });
    }

    static hipdnnPluginStatus_t
        enginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                         hipdnnPluginConstData_t* engineDetails)
    {
        LOG_API_ENTRY("handle={:p}, engineDetails={}",
                      static_cast<void*>(handle),
                      static_cast<void*>(engineDetails));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(handle);
            hipdnn_plugin::throwIfNull(engineDetails);

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           "No engine details to destroy");
            }

            hipdnn_plugin::throwIfNull(engineDetails->ptr);

            delete[] static_cast<const uint8_t*>(engineDetails->ptr);

            LOG_API_SUCCESS(apiName, "engineDetails->ptr={:p}", engineDetails->ptr);
        });
    }

    static hipdnnPluginStatus_t
        enginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                     const hipdnnPluginConstData_t* engineConfig,
                                     const hipdnnPluginConstData_t* opGraph,
                                     size_t* workspaceSize)
    {
        LOG_API_ENTRY("handle={:p}, engineConfig={:p}, opGraph={:p}, workspaceSize={:p}",
                      static_cast<void*>(handle),
                      static_cast<const void*>(engineConfig),
                      static_cast<const void*>(opGraph),
                      static_cast<void*>(workspaceSize));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(handle);
            hipdnn_plugin::throwIfNull(engineConfig);
            hipdnn_plugin::throwIfNull(opGraph);
            hipdnn_plugin::throwIfNull(workspaceSize);
            hipdnn_plugin::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot get workspace size");
            }

            *workspaceSize = 1024;

            LOG_API_SUCCESS(apiName, "workspaceSize={}", *workspaceSize);
        });
    }

    static hipdnnPluginStatus_t
        enginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t executionContext,
                                     size_t* workspaceSize)
    {
        LOG_API_ENTRY("handle={:p}, executionContext={:p}, workspaceSize={:p}",
                      static_cast<void*>(handle),
                      static_cast<const void*>(executionContext),
                      static_cast<void*>(workspaceSize));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(handle);
            hipdnn_plugin::throwIfNull(executionContext);
            hipdnn_plugin::throwIfNull(workspaceSize);
            hipdnn_plugin::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot get workspace size");
            }

            *workspaceSize = 2048;

            LOG_API_SUCCESS(apiName, "workspaceSize={}", *workspaceSize);
        });
    }

    static hipdnnPluginStatus_t
        enginePluginCreateExecutionContext(hipdnnEnginePluginHandle_t handle,
                                           const hipdnnPluginConstData_t* engineConfig,
                                           const hipdnnPluginConstData_t* opGraph,
                                           hipdnnEnginePluginExecutionContext_t* executionContext)
    {
        LOG_API_ENTRY("handle={:p}, engineConfig={:p}, opGraph={:p}, executionContext={:p}",
                      static_cast<void*>(handle),
                      static_cast<const void*>(engineConfig),
                      static_cast<const void*>(opGraph),
                      static_cast<void*>(executionContext));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(handle);
            hipdnn_plugin::throwIfNull(engineConfig);
            hipdnn_plugin::throwIfNull(opGraph);
            hipdnn_plugin::throwIfNull(executionContext);
            hipdnn_plugin::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot create execution context");
            }

            hipdnn_plugin::GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);
            hipdnn_plugin::EngineConfigWrapper engineConfigWrapper(engineConfig->ptr,
                                                                   engineConfig->size);

            *executionContext = new HipdnnEnginePluginExecutionContext();

            LOG_API_SUCCESS(
                apiName, "createdExecutionContext={:p}", static_cast<void*>(*executionContext));
        });
    }

    static hipdnnPluginStatus_t
        enginePluginDestroyExecutionContext(hipdnnEnginePluginHandle_t handle,
                                            hipdnnEnginePluginExecutionContext_t executionContext)
    {
        LOG_API_ENTRY("handle={:p}, executionContext={:p}",
                      static_cast<void*>(handle),
                      static_cast<void*>(executionContext));

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(handle);
            hipdnn_plugin::throwIfNull(executionContext);
            hipdnn_plugin::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           "No execution context to destroy");
            }

            delete executionContext;

            LOG_API_SUCCESS(apiName, "destroyed executionContext", "");
        });
    }

    static hipdnnPluginStatus_t
        enginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                   hipdnnEnginePluginExecutionContext_t executionContext,
                                   void* workspace,
                                   const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                   uint32_t numDeviceBuffers)
    {
        LOG_API_ENTRY("handle={:p}, executionContext={:p}, workspace={:p}, deviceBuffers={:p}, "
                      "numDeviceBuffers={}",
                      static_cast<void*>(handle),
                      static_cast<void*>(executionContext),
                      workspace,
                      static_cast<const void*>(deviceBuffers),
                      numDeviceBuffers);

        return hipdnn_plugin::tryCatch([&, apiName = __func__]() {
            hipdnn_plugin::throwIfNull(handle);
            hipdnn_plugin::throwIfNull(executionContext);
            hipdnn_plugin::throwIfNull(deviceBuffers);
            hipdnn_plugin::throwIfNull(getInstance());

            if(!getInstance()->supportsEngineOperations())
            {
                throw hipdnn_plugin::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    "No engines available - cannot execute graph");
            }

            getInstance()->executeGraph();

            LOG_API_SUCCESS(apiName, "executed graph", "");
        });
    }

private:
    inline static std::unique_ptr<TestPluginBase> s_instance; //NOLINT
};

// Macro to register plugin API functions
#define REGISTER_TEST_PLUGIN_API()                                                                \
    extern "C" {                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)                                   \
    {                                                                                             \
        return TestPluginBase::pluginGetName(name);                                               \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)                             \
    {                                                                                             \
        return TestPluginBase::pluginGetVersion(version);                                         \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)                            \
    {                                                                                             \
        return TestPluginBase::pluginGetType(type);                                               \
    }                                                                                             \
                                                                                                  \
    void hipdnnPluginGetLastErrorString(const char** errorStr)                                    \
    {                                                                                             \
        TestPluginBase::pluginGetLastErrorString(errorStr);                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)                \
    {                                                                                             \
        return TestPluginBase::pluginSetLoggingCallback(callback);                                \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds,                    \
                                                           uint32_t maxEngines,                   \
                                                           uint32_t* numEngines)                  \
    {                                                                                             \
        return TestPluginBase::enginePluginGetAllEngineIds(engineIds, maxEngines, numEngines);    \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)             \
    {                                                                                             \
        return TestPluginBase::enginePluginCreate(handle);                                        \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)             \
    {                                                                                             \
        return TestPluginBase::enginePluginDestroy(handle);                                       \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,           \
                                                     hipStream_t stream)                          \
    {                                                                                             \
        return TestPluginBase::enginePluginSetStream(handle, stream);                             \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,               \
                                                 const hipdnnPluginConstData_t* opGraph,          \
                                                 int64_t* engineIds,                              \
                                                 uint32_t maxEngines,                             \
                                                 uint32_t* numEngines)                            \
    {                                                                                             \
        return TestPluginBase::enginePluginGetApplicableEngineIds(                                \
            handle, opGraph, engineIds, maxEngines, numEngines);                                  \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,                     \
                                           int64_t engineId,                                      \
                                           const hipdnnPluginConstData_t* opGraph,                \
                                           hipdnnPluginConstData_t* engineDetails)                \
    {                                                                                             \
        return TestPluginBase::enginePluginGetEngineDetails(                                      \
            handle, engineId, opGraph, engineDetails);                                            \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,                 \
                                               hipdnnPluginConstData_t* engineDetails)            \
    {                                                                                             \
        return TestPluginBase::enginePluginDestroyEngineDetails(handle, engineDetails);           \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,                     \
                                           const hipdnnPluginConstData_t* engineConfig,           \
                                           const hipdnnPluginConstData_t* opGraph,                \
                                           size_t* workspaceSize)                                 \
    {                                                                                             \
        return TestPluginBase::enginePluginGetWorkspaceSize(                                      \
            handle, engineConfig, opGraph, workspaceSize);                                        \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(                  \
        hipdnnEnginePluginHandle_t handle,                                                        \
        hipdnnEnginePluginExecutionContext_t executionContext,                                    \
        size_t* workspaceSize)                                                                    \
    {                                                                                             \
        return TestPluginBase::enginePluginGetWorkspaceSize(                                      \
            handle, executionContext, workspaceSize);                                             \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContext(                                \
        hipdnnEnginePluginHandle_t handle,                                                        \
        const hipdnnPluginConstData_t* engineConfig,                                              \
        const hipdnnPluginConstData_t* opGraph,                                                   \
        hipdnnEnginePluginExecutionContext_t* executionContext)                                   \
    {                                                                                             \
        return TestPluginBase::enginePluginCreateExecutionContext(                                \
            handle, engineConfig, opGraph, executionContext);                                     \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(                               \
        hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t executionContext) \
    {                                                                                             \
        return TestPluginBase::enginePluginDestroyExecutionContext(handle, executionContext);     \
    }                                                                                             \
                                                                                                  \
    hipdnnPluginStatus_t                                                                          \
        hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,                       \
                                         hipdnnEnginePluginExecutionContext_t executionContext,   \
                                         void* workspace,                                         \
                                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,         \
                                         uint32_t numDeviceBuffers)                               \
    {                                                                                             \
        return TestPluginBase::enginePluginExecuteOpGraph(                                        \
            handle, executionContext, workspace, deviceBuffers, numDeviceBuffers);                \
    }                                                                                             \
    } // extern "C"
