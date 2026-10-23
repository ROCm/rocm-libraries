// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

namespace hipdnn_backend::heuristics::prediction
{

hipdnnPluginStatus_t getName(const char** name);
hipdnnPluginStatus_t getVersion(const char** version);
hipdnnPluginStatus_t getApiVersion(const char** version);
hipdnnPluginStatus_t getType(hipdnnPluginType_t* type);
hipdnnPluginStatus_t setLoggingCallback(hipdnnCallback_t callback);
hipdnnPluginStatus_t setLogLevel(hipdnnSeverity_t level);
void getLastErrorString(const char** error);
hipdnnPluginStatus_t getAllPolicyIds(int64_t* ids, uint32_t capacity, uint32_t* count);
hipdnnPluginStatus_t getPolicyName(int64_t id, const char** name);
hipdnnPluginStatus_t handleCreate(hipdnnHeuristicHandle_t* handle);
hipdnnPluginStatus_t handleDestroy(hipdnnHeuristicHandle_t handle);
hipdnnPluginStatus_t handleSetDeviceProperties(hipdnnHeuristicHandle_t handle,
                                               const hipdnnPluginConstData_t* properties);
hipdnnPluginStatus_t policyDescriptorCreate(hipdnnHeuristicHandle_t handle,
                                            int64_t policyId,
                                            hipdnnHeuristicPolicyDescriptor_t* descriptor);
hipdnnPluginStatus_t policyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t descriptor);
hipdnnPluginStatus_t policySetEngineIds(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                        const int64_t* ids,
                                        size_t count);
hipdnnPluginStatus_t policySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                              const hipdnnPluginConstData_t* graph);
hipdnnPluginStatus_t policyFinalize(hipdnnHeuristicPolicyDescriptor_t descriptor, int32_t* applied);
hipdnnPluginStatus_t policyFinalizeWithHost(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                            const hipdnnHeuristicHostCallbacks_t* host,
                                            int32_t* applied);
hipdnnPluginStatus_t policyGetSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                              int64_t* ids,
                                              size_t* count);
hipdnnPluginStatus_t policyGetEngineConfig(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                           int64_t engineId,
                                           hipdnnPluginConstData_t* config);

} // namespace hipdnn_backend::heuristics::prediction
