// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/backend/HipdnnBackendInterface.hpp>

namespace hipdnn_frontend
{

class HipdnnBackendWrapper : public IHipdnnBackend
{
public:
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

    hipdnnStatus_t backendCreateAndDeserializeGraphExt(hipdnnBackendDescriptor_t* descriptor,
                                                       const uint8_t* serializedGraph,
                                                       size_t graphByteSize) override
    {
        return hipdnnBackendCreateAndDeserializeGraph_ext(
            descriptor, serializedGraph, graphByteSize);
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
};

// Allow overriding the backend implementation by setting a custom backend instance.
inline static std::shared_ptr<IHipdnnBackend> hipdnnBackend()
{
    if(!IHipdnnBackend::getInstance())
    {
        IHipdnnBackend::setInstance(std::make_shared<HipdnnBackendWrapper>());
    }

    return IHipdnnBackend::getInstance();
}

}
