// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file is part of the test plugin implementation.

#pragma once

#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>

void getAllEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t* numEngines);
void getApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                            const hipdnnPluginConstData_t* opGraph,
                            int64_t* engineIds,
                            uint32_t maxEngines,
                            uint32_t* numEngines);
void checkEngineIdValidity(int64_t engineId); // throw if invalid
void getEngineDetails(hipdnnEnginePluginHandle_t handle,
                      int64_t engineId,
                      const hipdnnPluginConstData_t* opGraph,
                      hipdnnPluginConstData_t* engineDetails);
void destroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                          hipdnnPluginConstData_t* engineDetails);
size_t getWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                        const hipdnnPluginConstData_t* engineConfig,
                        const hipdnnPluginConstData_t* opGraph);
size_t getWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                        hipdnnEnginePluginExecutionContext_t executionContext);
hipdnnEnginePluginExecutionContext_t
    createExecutionContext(hipdnnEnginePluginHandle_t handle,
                           const hipdnnPluginConstData_t* engineConfig,
                           const hipdnnPluginConstData_t* opGraph);
void destroyExecutionContext(hipdnnEnginePluginHandle_t handle,
                             hipdnnEnginePluginExecutionContext_t executionContext);
void executeOpGraph(hipdnnEnginePluginHandle_t handle,
                    hipdnnEnginePluginExecutionContext_t executionContext,
                    void* workspace,
                    const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                    uint32_t numDeviceBuffers);
