// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/detail/HipdnnBackendInterface.hpp>
#include <hipdnn_frontend/detail/IncompatibleBackend.hpp>
#include <hipdnn_frontend/version.h>

namespace hipdnn_frontend::detail
{

class HipdnnBackendWrapper : public IHipdnnBackend
{
public:
    HipdnnBackendWrapper(hipdnn_data_sdk::utilities::Version version)
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

    hipdnnStatus_t getLoadedEnginePluginPathsExt(hipdnnHandle_t handle,
                                                 size_t* numPluginPaths,
                                                 char** pluginPaths,
                                                 size_t* maxStringLen) override
    {
        return hipdnnGetLoadedEnginePluginPaths_ext(
            handle, numPluginPaths, pluginPaths, maxStringLen);
    }

    // RFC 0007 Section 16: Heuristic policy enumeration
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

private:
    hipdnn_data_sdk::utilities::Version _version;
};

// Attempts to create a backend interface of type T, falling back to IncompatibleBackend if it fails to satisfy requirements
// version is taken as an argument to facilitate easier testing
inline std::shared_ptr<IHipdnnBackend> tryToUseBackendInterface(const char* version)
{
    using namespace hipdnn_data_sdk::utilities;

    if(version == nullptr)
    {
        HIPDNN_FE_LOG_ERROR("Error parsing backend version: version is nullptr");
        return std::make_shared<IncompatibleBackendWrapper>();
    }

    Version backendVersion;
    try
    {
        backendVersion = Version{std::string{version}};
    }
    catch(const std::invalid_argument& error)
    {
        HIPDNN_FE_LOG_ERROR("Error parsing backend version: " + std::string{error.what()});
        return std::make_shared<IncompatibleBackendWrapper>();
    }

    if(HIPDNN_FRONTEND_VERSION_MAJOR != backendVersion.major)
    {
        HIPDNN_FE_LOG_ERROR("Backend major version (" + std::to_string(backendVersion.major)
                            + ") does not match frontend major version ("
                            + std::to_string(HIPDNN_FRONTEND_VERSION_MAJOR) + ")");
        return std::make_shared<IncompatibleBackendWrapper>();
    }

    return std::make_shared<HipdnnBackendWrapper>(backendVersion);
}

// Allow overriding the backend implementation by setting a custom backend instance.
inline static std::shared_ptr<IHipdnnBackend> hipdnnBackend()
{
    if(!IHipdnnBackend::getInstance())
    {
        IHipdnnBackend::setInstance(tryToUseBackendInterface(hipdnnVersionString_ext()));
    }

    return IHipdnnBackend::getInstance();
}

} // namespace hipdnn_frontend::detail
