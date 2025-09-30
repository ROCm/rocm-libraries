// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_backend.h"
#include <hipdnn_sdk/data_objects/graph_generated.h>

namespace test_util
{

void createTestHandle(hipdnnHandle_t* handle);

void createTestGraph(hipdnnBackendDescriptor_t* descriptor, hipdnnHandle_t handle);

void populateTestEngine(hipdnnBackendDescriptor_t engine,
                        hipdnnBackendDescriptor_t* graph,
                        hipdnnHandle_t handle,
                        int64_t gidx,
                        bool finalize = false);

void createTestEngine(hipdnnBackendDescriptor_t* engine,
                      hipdnnBackendDescriptor_t* graph,
                      hipdnnHandle_t handle,
                      int64_t gidx,
                      bool finalize = false);

void populateTestEngineConfig(hipdnnBackendDescriptor_t* engineConfig,
                              hipdnnBackendDescriptor_t* engine,
                              hipdnnBackendDescriptor_t* graph,
                              hipdnnHandle_t handle,
                              int64_t gidx,
                              bool finalize = false);

void createTestEngineConfig(hipdnnBackendDescriptor_t* engineConfig,
                            hipdnnBackendDescriptor_t* engine,
                            hipdnnBackendDescriptor_t* graph,
                            hipdnnHandle_t handle,
                            int64_t gidx,
                            bool finalize = false);

void populateTestExecutionPlan(hipdnnBackendDescriptor_t* executionPlan,
                               hipdnnBackendDescriptor_t* engineConfig,
                               hipdnnBackendDescriptor_t* engine,
                               hipdnnBackendDescriptor_t* graph,
                               hipdnnHandle_t handle,
                               int64_t gidx,
                               bool finalize = false);

flatbuffers::FlatBufferBuilder createAndPopulateBatchnormNode();

void* allocateTensorMemory(const int64_t* dims,
                           size_t dimsCount,
                           hipdnnBackendAttributeType_t dataType,
                           bool initialize);

void freeTensorMemory(void* dataPtr);

void populateVariantPackWithMappings(hipdnnBackendDescriptor_t variantPack,
                                     const std::unordered_map<int64_t, void*>& dataPtrMappings,
                                     void* workspace = nullptr);

void createAndInitializeBackendDescriptor(hipdnnBackendDescriptor_t* backendDescriptor,
                                          const flatbuffers::DetachedBuffer& serializedGraph,
                                          hipdnnHandle_t handle);

void extractTensorInfoFromGraph(const flatbuffers::DetachedBuffer& serializedGraph,
                                std::unordered_map<int64_t, std::string>& uidToNameMap,
                                std::unordered_map<std::string, int64_t>& nameToUidMap,
                                std::unordered_map<int64_t, std::vector<int64_t>>& uidToDimsMap);

std::vector<std::string> getLoadedPlugins(hipdnnHandle_t handle);

bool isPluginLoaded(const std::vector<std::string>& loadedPlugins, const std::string& pluginName);

bool isPluginLoadedByRelativePath(const std::vector<std::string>& loadedPlugins,
                                  const std::string& relativePath);

} // namespace test_util
