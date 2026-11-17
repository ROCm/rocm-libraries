// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_frontend/detail/HipdnnBackendInterface.hpp>

namespace hipdnn_frontend::detail
{

#ifndef HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND

class HipdnnDirectBackendWrapper : public IHipdnnBackend
{
public:
    HipdnnDirectBackendWrapper(hipdnn_data_sdk::utilities::Version version)
        : _version(version)
    {
    }
    hipdnnStatus_t create(hipdnnHandle_t* handle) override
    {
        return hipdnnCreate(handle);
    }

    hipdnnStatus_t destroy(hipdnnHandle_t handle) override
    {
        return hipdnnDestroy(handle);
    }

    hipdnnStatus_t setStream(hipdnnHandle_t handle, hipStream_t streamId) override
    {
        return hipdnnSetStream(handle, streamId);
    }

    hipdnnStatus_t getStream(hipdnnHandle_t handle, hipStream_t* streamId) override
    {
        return hipdnnGetStream(handle, streamId);
    }

    hipdnnStatus_t backendCreateDescriptor(hipdnnBackendDescriptorType_t descriptorType,
                                           hipdnnBackendDescriptor_t* descriptor) override
    {
        return hipdnnBackendCreateDescriptor(descriptorType, descriptor);
    }

    hipdnnStatus_t backendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor) override
    {
        return hipdnnBackendDestroyDescriptor(descriptor);
    }

    hipdnnStatus_t backendExecute(hipdnnHandle_t handle,
                                  hipdnnBackendDescriptor_t executionPlan,
                                  hipdnnBackendDescriptor_t variantPack) override
    {
        return hipdnnBackendExecute(handle, executionPlan, variantPack);
    }

    hipdnnStatus_t backendFinalize(hipdnnBackendDescriptor_t descriptor) override
    {
        return hipdnnBackendFinalize(descriptor);
    }

    hipdnnStatus_t backendGetAttribute(hipdnnBackendDescriptor_t descriptor,
                                       hipdnnBackendAttributeName_t attributeName,
                                       hipdnnBackendAttributeType_t attributeType,
                                       int64_t requestedElementCount,
                                       int64_t* elementCount,
                                       void* arrayOfElements) override
    {
        return hipdnnBackendGetAttribute(descriptor,
                                         attributeName,
                                         attributeType,
                                         requestedElementCount,
                                         elementCount,
                                         arrayOfElements);
    }

    hipdnnStatus_t backendSetAttribute(hipdnnBackendDescriptor_t descriptor,
                                       hipdnnBackendAttributeName_t attributeName,
                                       hipdnnBackendAttributeType_t attributeType,
                                       int64_t elementCount,
                                       const void* arrayOfElements) override
    {
        return hipdnnBackendSetAttribute(
            descriptor, attributeName, attributeType, elementCount, arrayOfElements);
    }

    const char* getErrorString(hipdnnStatus_t status) override
    {
        return hipdnnGetErrorString(status);
    }

    void getLastErrorString(char* message, size_t maxSize) override
    {
        hipdnnGetLastErrorString(message, maxSize);
    }

    hipdnn_data_sdk::utilities::Version version() override
    {
        return _version;
    }

    const char* versionString() override
    {
        return hipdnnVersionString_ext();
    }

    hipdnnStatus_t backendCreateAndDeserializeGraphExt(hipdnnBackendDescriptor_t* descriptor,
                                                       const uint8_t* serializedGraph,
                                                       size_t graphByteSize) override
    {
        return hipdnnBackendCreateAndDeserializeGraph_ext(
            descriptor, serializedGraph, graphByteSize);
    }

    hipdnnStatus_t backendGetSerializedBinaryGraphExt(hipdnnBackendDescriptor_t descriptor,
                                                      size_t requestedByteSize,
                                                      size_t* graphByteSize,
                                                      uint8_t* serializedGraph) override
    {
        return hipdnnBackendGetSerializedBinaryGraph_ext(
            descriptor, requestedByteSize, graphByteSize, serializedGraph);
    }

    hipdnnStatus_t backendGetSerializedJsonGraphExt(hipdnnBackendDescriptor_t descriptor,
                                                    size_t requestedByteSize,
                                                    size_t* graphByteSize,
                                                    char* serializedJsonGraph) override
    {
        return hipdnnBackendGetSerializedJsonGraph_ext(
            descriptor, requestedByteSize, graphByteSize, serializedJsonGraph);
    }

    hipdnnStatus_t backendCreateAndDeserializeJsonGraphExt(hipdnnBackendDescriptor_t* descriptor,
                                                           const char* jsonGraph,
                                                           size_t jsonByteSize) override
    {
        return hipdnnBackendCreateAndDeserializeJsonGraph_ext(descriptor, jsonGraph, jsonByteSize);
    }

    hipdnnStatus_t backendGetSerializedExecutionPlanExt(hipdnnBackendDescriptor_t descriptor,
                                                        size_t requestedByteSize,
                                                        size_t* planByteSize,
                                                        uint8_t* serializedPlan) override
    {
        return hipdnnBackendGetSerializedExecutionPlan_ext(
            descriptor, requestedByteSize, planByteSize, serializedPlan);
    }

    hipdnnStatus_t
        backendCreateAndDeserializeExecutionPlanExt(hipdnnHandle_t handle,
                                                    hipdnnBackendDescriptor_t* descriptor,
                                                    const uint8_t* serializedPlan,
                                                    size_t planByteSize) override
    {
        return hipdnnBackendCreateAndDeserializeExecutionPlan_ext(
            handle, descriptor, serializedPlan, planByteSize);
    }

    hipdnnStatus_t
        backendGetSerializedBinaryGraphAndPlanExt(hipdnnBackendDescriptor_t graphDescriptor,
                                                  hipdnnBackendDescriptor_t executionPlanDescriptor,
                                                  size_t requestedByteSize,
                                                  size_t* blobByteSize,
                                                  uint8_t* serializedBlob) override
    {
        return hipdnnBackendGetSerializedBinaryGraphAndPlan_ext(graphDescriptor,
                                                                executionPlanDescriptor,
                                                                requestedByteSize,
                                                                blobByteSize,
                                                                serializedBlob);
    }

    hipdnnStatus_t backendGetSerializedBinaryContentsExt(const uint8_t* serializedBlob,
                                                         size_t blobByteSize,
                                                         int* contentFlags) override
    {
        return hipdnnBackendGetSerializedBinaryContents_ext(
            serializedBlob, blobByteSize, contentFlags);
    }

    void loggingCallbackExt(hipdnnSeverity_t severity, const char* msg) override
    {
        hipdnnLoggingCallback_ext(severity, msg);
    }

    hipdnnStatus_t setEnginePluginPathsExt(size_t numPaths,
                                           const char* const* pluginPaths,
                                           hipdnnPluginLoadingMode_ext_t mode) override
    {
        return hipdnnSetEnginePluginPaths_ext(numPaths, pluginPaths, mode);
    }

    hipdnnStatus_t setHeuristicPluginPathsExt(size_t numPaths,
                                              const char* const* pluginPaths,
                                              hipdnnPluginLoadingMode_ext_t mode) override
    {
        return hipdnnSetHeuristicPluginPaths_ext(numPaths, pluginPaths, mode);
    }

    hipdnnStatus_t getLoadedEnginePluginPathsExt(hipdnnHandle_t handle,
                                                 size_t* numPluginPaths,
                                                 char** pluginPaths,
                                                 size_t* maxStringLen) override
    {
        return hipdnnGetLoadedEnginePluginPaths_ext(
            handle, numPluginPaths, pluginPaths, maxStringLen);
    }

    hipdnnStatus_t getEngineNameByIdExt(hipdnnHandle_t handle,
                                        int64_t engineId,
                                        char* engineName,
                                        size_t* engineNameLen) override
    {
        return hipdnnGetEngineNameById_ext(handle, engineId, engineName, engineNameLen);
    }

    hipdnnStatus_t getHeuristicPolicyCount(hipdnnHandle_t handle, size_t* numPolicies) override
    {
        return hipdnnGetHeuristicPolicyCount_ext(handle, numPolicies);
    }

    hipdnnStatus_t getHeuristicPolicyInfo(hipdnnHandle_t handle,
                                          size_t policyIndex,
                                          int64_t* policyId,
                                          char* policyName,
                                          size_t* policyNameLen,
                                          char* pluginName,
                                          size_t* pluginNameLen,
                                          char* pluginVersion,
                                          size_t* pluginVersionLen,
                                          char* apiVersion,
                                          size_t* apiVersionLen) override
    {
        return hipdnnGetHeuristicPolicyInfo_ext(handle,
                                                policyIndex,
                                                policyId,
                                                policyName,
                                                policyNameLen,
                                                pluginName,
                                                pluginNameLen,
                                                pluginVersion,
                                                pluginVersionLen,
                                                apiVersion,
                                                apiVersionLen);
    }

    hipdnnStatus_t setUserLogCallbackExt(hipdnnUserLogCallback_t callback,
                                         hipdnnSeverity_t minLevel,
                                         hipdnnLogCallbackMode_t mode,
                                         hipdnnUserLogCallbackHandle_t userHandle) override
    {
        return hipdnnSetUserLogCallback_ext(callback, minLevel, mode, userHandle);
    }

    hipdnnStatus_t backendSetGlobalLogLevelExt(hipdnnSeverity_t level) override
    {
        return hipdnnBackendSetGlobalLogLevel_ext(level);
    }

    hipdnnStatus_t backendGetGlobalLogLevelExt(hipdnnSeverity_t* level) override
    {
        return hipdnnBackendGetGlobalLogLevel_ext(level);
    }

    hipdnnStatus_t writeEngineRankingResultsExt(hipdnnHandle_t handle,
                                                hipdnnBackendDescriptor_t graphDescriptor,
                                                const int64_t* engineIdsInRankOrder,
                                                size_t engineIdCount,
                                                hipdnnAutotuneCacheWriteOutcome_ext_t* outcome
                                                = nullptr) override
    {
        return hipdnnBackendWriteEngineRankingResults_ext(
            handle, graphDescriptor, engineIdsInRankOrder, engineIdCount, outcome);
    }

private:
    hipdnn_data_sdk::utilities::Version _version;
};

#endif // !HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND

} // namespace hipdnn_frontend::detail
