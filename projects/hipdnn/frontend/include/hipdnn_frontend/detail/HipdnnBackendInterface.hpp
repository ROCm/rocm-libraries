// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <mutex>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/Visibility.hpp>

namespace hipdnn_frontend::detail
{

class IHipdnnBackend
{
public:
    virtual ~IHipdnnBackend() = default;

    virtual hipdnnStatus_t create(hipdnnHandle_t* handle) = 0;
    virtual hipdnnStatus_t destroy(hipdnnHandle_t handle) = 0;
    virtual hipdnnStatus_t setStream(hipdnnHandle_t handle, hipStream_t streamId) = 0;
    virtual hipdnnStatus_t getStream(hipdnnHandle_t handle, hipStream_t* streamId) = 0;
    virtual hipdnnStatus_t backendCreateDescriptor(hipdnnBackendDescriptorType_t descriptorType,
                                                   hipdnnBackendDescriptor_t* descriptor)
        = 0;
    virtual hipdnnStatus_t backendDestroyDescriptor(hipdnnBackendDescriptor_t descriptor) = 0;
    virtual hipdnnStatus_t backendExecute(hipdnnHandle_t handle,
                                          hipdnnBackendDescriptor_t executionPlan,
                                          hipdnnBackendDescriptor_t variantPack)
        = 0;
    virtual hipdnnStatus_t backendFinalize(hipdnnBackendDescriptor_t descriptor) = 0;
    virtual hipdnnStatus_t backendGetAttribute(hipdnnBackendDescriptor_t descriptor,
                                               hipdnnBackendAttributeName_t attributeName,
                                               hipdnnBackendAttributeType_t attributeType,
                                               int64_t requestedElementCount,
                                               int64_t* elementCount,
                                               void* arrayOfElements)
        = 0;
    virtual hipdnnStatus_t backendSetAttribute(hipdnnBackendDescriptor_t descriptor,
                                               hipdnnBackendAttributeName_t attributeName,
                                               hipdnnBackendAttributeType_t attributeType,
                                               int64_t elementCount,
                                               const void* arrayOfElements)
        = 0;
    virtual const char* getErrorString(hipdnnStatus_t status) = 0;
    virtual void getLastErrorString(char* message, size_t maxSize) = 0;
    virtual hipdnnStatus_t versionExt(const char** version) = 0;
    virtual hipdnnStatus_t backendCreateAndDeserializeGraphExt(
        hipdnnBackendDescriptor_t* descriptor, const uint8_t* serializedGraph, size_t graphByteSize)
        = 0;
    virtual void loggingCallbackExt(hipdnnSeverity_t severity, const char* msg) = 0;

    virtual hipdnnStatus_t setEnginePluginPathsExt(size_t numPaths,
                                                   const char* const* pluginPaths,
                                                   hipdnnPluginLoadingMode_ext_t mode)
        = 0;

    // HIPDNN_HIDDEN on accessor functions ensures each shared object has its own backendInstance
    static inline std::shared_ptr<IHipdnnBackend> backendInstance;
    HIPDNN_HIDDEN static std::shared_ptr<IHipdnnBackend> getInstance()
    {
        return backendInstance;
    }

    HIPDNN_HIDDEN static void setInstance(std::shared_ptr<IHipdnnBackend> instance)
    {
        backendInstance = std::move(instance);
    }

    HIPDNN_HIDDEN static void resetInstance()
    {
        backendInstance.reset();
    }
};

} // namespace hipdnn_frontend::detail
