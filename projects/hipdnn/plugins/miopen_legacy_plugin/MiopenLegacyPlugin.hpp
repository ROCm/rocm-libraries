// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/plugin/PluginApi.h>

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetNamePvt(const char** name);

hipdnnPluginStatus_t hipdnnPluginGetVersionPvt(const char** version);
hipdnnPluginStatus_t hipdnnPluginGetTypePvt(hipdnnPluginType_t* type);
void hipdnnPluginGetLastErrorStringPvt(const char** errorStr);
hipdnnPluginStatus_t hipdnnPluginSetLoggingCallbackPvt(hipdnnCallback_t callback);
hipdnnPluginStatus_t hipdnnEnginePluginGetAllEngineIdsPvt(int64_t* engineIds,
                                                          uint32_t maxEngines,
                                                          uint32_t* numEngines);
hipdnnPluginStatus_t hipdnnEnginePluginCreatePvt(hipdnnEnginePluginHandle_t* handle);
hipdnnPluginStatus_t hipdnnEnginePluginDestroyPvt(hipdnnEnginePluginHandle_t handle);
hipdnnPluginStatus_t hipdnnEnginePluginSetStreamPvt(hipdnnEnginePluginHandle_t handle,
                                                    hipStream_t stream);
hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIdsPvt(hipdnnEnginePluginHandle_t handle,
                                                const hipdnnPluginConstData_t* opGraph,
                                                int64_t* engineIds,
                                                uint32_t maxEngines,
                                                uint32_t* numEngines);
hipdnnPluginStatus_t hipdnnEnginePluginGetEngineDetailsPvt(hipdnnEnginePluginHandle_t handle,
                                                           int64_t engineId,
                                                           const hipdnnPluginConstData_t* opGraph,
                                                           hipdnnPluginConstData_t* engineDetails);
hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyEngineDetailsPvt(hipdnnEnginePluginHandle_t handle,
                                              hipdnnPluginConstData_t* engineDetails);
hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSizePvt(hipdnnEnginePluginHandle_t handle,
                                          const hipdnnPluginConstData_t* engineConfig,
                                          const hipdnnPluginConstData_t* opGraph,
                                          size_t* workspaceSize);
hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContextPvt(
    hipdnnEnginePluginHandle_t handle,
    const hipdnnPluginConstData_t* engineConfig,
    const hipdnnPluginConstData_t* opGraph,
    hipdnnEnginePluginExecutionContext_t* executionContext);
hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContextPvt(
    hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t executionContext);
hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSizeFromExecutionContextPvt(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext,
    size_t* workspaceSize);
hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraphPvt(hipdnnEnginePluginHandle_t handle,
                                        hipdnnEnginePluginExecutionContext_t executionContext,
                                        void* workspace,
                                        const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                        uint32_t numDeviceBuffers);

} // extern "C"
