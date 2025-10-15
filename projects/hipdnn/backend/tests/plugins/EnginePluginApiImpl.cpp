// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file is part of the test plugin implementation.
// It contains the API functions for the test engine plugin.

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/plugin/PluginHelpers.hpp>

#include "EnginePluginApiImpl.hpp"
#include "EnginePluginHandle.hpp"

using namespace hipdnn_plugin;

#ifdef THROW_IF_NULL
#error "THROW_IF_NULL is already defined"
#endif
#define THROW_IF_NULL(value) \
    PLUGIN_THROW_IF_NULL(value, HIPDNN_PLUGIN_STATUS_BAD_PARAM, #value " is null")

#ifdef THROW_IF_EQ
#error "THROW_IF_EQ is already defined"
#endif
#define THROW_IF_EQ(value, expected) \
    PLUGIN_THROW_IF_EQ(value, expected, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, #value " is " #expected)

namespace
{

void checkHandleValidity(hipdnnEnginePluginHandle_t handle)
{
    THROW_IF_NULL(handle);
}

} // namespace

// Exported functions:

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginGetAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines)
{
    return hipdnn_plugin::tryCatch([&]() {
        if(maxEngines != 0)
        {
            THROW_IF_NULL(engineIds);
        }
        THROW_IF_NULL(numEngines);

        getAllEngineIds(engineIds, maxEngines, numEngines);
    });
}

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)
{
    return hipdnn_plugin::tryCatch([&]() {
        THROW_IF_NULL(handle);
        *handle = new HipdnnEnginePluginHandle{nullptr};
    });
}

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)
{
    return hipdnn_plugin::tryCatch([&]() {
        checkHandleValidity(handle);
        delete handle;
    });
}

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                                                            hipStream_t stream)
{
    return hipdnn_plugin::tryCatch([&]() {
        checkHandleValidity(handle);
        handle->stream = stream;
    });
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* opGraph,
                                             int64_t* engineIds,
                                             uint32_t maxEngines,
                                             uint32_t* numEngines)
{
    return hipdnn_plugin::tryCatch([&]() {
        checkHandleValidity(handle);
        THROW_IF_NULL(opGraph);
        if(maxEngines != 0)
        {
            THROW_IF_NULL(engineIds);
        }
        THROW_IF_NULL(numEngines);

        getApplicableEngineIds(handle, opGraph, engineIds, maxEngines, numEngines);
    });
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                       int64_t engineId,
                                       const hipdnnPluginConstData_t* opGraph,
                                       hipdnnPluginConstData_t* engineDetails)
{
    return hipdnn_plugin::tryCatch([&]() {
        checkHandleValidity(handle);
        checkEngineIdValidity(engineId);
        THROW_IF_NULL(opGraph);
        THROW_IF_NULL(engineDetails);

        getEngineDetails(handle, engineId, opGraph, engineDetails);
    });
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                           hipdnnPluginConstData_t* engineDetails)
{
    return hipdnn_plugin::tryCatch([&]() {
        checkHandleValidity(handle);
        THROW_IF_NULL(engineDetails);

        destroyEngineDetails(handle, engineDetails);
    });
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                       const hipdnnPluginConstData_t* engineConfig,
                                       const hipdnnPluginConstData_t* opGraph,
                                       size_t* workspaceSize)
{
    return hipdnn_plugin::tryCatch([&]() {
        checkHandleValidity(handle);
        THROW_IF_NULL(engineConfig);
        THROW_IF_NULL(opGraph);
        THROW_IF_NULL(workspaceSize);

        *workspaceSize = getWorkspaceSize(handle, engineConfig, opGraph);
    });
}

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext,
    size_t* workspaceSize)
{
    return hipdnn_plugin::tryCatch([&]() {
        checkHandleValidity(handle);
        THROW_IF_NULL(executionContext);
        THROW_IF_NULL(workspaceSize);

        *workspaceSize = getWorkspaceSize(handle, executionContext);
    });
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginCreateExecutionContext(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* engineConfig,
                                             const hipdnnPluginConstData_t* opGraph,
                                             hipdnnEnginePluginExecutionContext_t* executionContext)
{
    return hipdnn_plugin::tryCatch([&]() {
        checkHandleValidity(handle);
        THROW_IF_NULL(engineConfig);
        THROW_IF_NULL(opGraph);
        THROW_IF_NULL(executionContext);

        *executionContext = createExecutionContext(handle, engineConfig, opGraph);
    });
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyExecutionContext(hipdnnEnginePluginHandle_t handle,
                                              hipdnnEnginePluginExecutionContext_t executionContext)
{
    return hipdnn_plugin::tryCatch([&]() {
        checkHandleValidity(handle);
        THROW_IF_NULL(executionContext);

        destroyExecutionContext(handle, executionContext);
    });
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t executionContext,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                     uint32_t numDeviceBuffers)
{
    return hipdnn_plugin::tryCatch([&]() {
        checkHandleValidity(handle);
        THROW_IF_NULL(executionContext);
        // Workspace can be null if the workspace size is zero.
        THROW_IF_NULL(deviceBuffers);
        THROW_IF_EQ(numDeviceBuffers, 0);

        executeOpGraph(handle, executionContext, workspace, deviceBuffers, numDeviceBuffers);
    });
}
