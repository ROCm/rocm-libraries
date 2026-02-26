// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/detail/HipdnnBackendInterface.hpp>

namespace hipdnn_frontend::detail
{

class IncompatibleBackendWrapper : public IHipdnnBackend
{
public:
    hipdnnStatus_t create(hipdnnHandle_t* /*handle*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t destroy(hipdnnHandle_t /*handle*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t setStream(hipdnnHandle_t /*handle*/, hipStream_t /*streamId*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t getStream(hipdnnHandle_t /*handle*/, hipStream_t* /*streamId*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t backendCreateDescriptor(hipdnnBackendDescriptorType_t /*descriptorType*/,
                                           hipdnnBackendDescriptor_t* /*descriptor*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t backendDestroyDescriptor(hipdnnBackendDescriptor_t /*descriptor*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t backendExecute(hipdnnHandle_t /*handle*/,
                                  hipdnnBackendDescriptor_t /*executionPlan*/,
                                  hipdnnBackendDescriptor_t /*variantPack*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t backendFinalize(hipdnnBackendDescriptor_t /*descriptor*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t backendGetAttribute(hipdnnBackendDescriptor_t /*descriptor*/,
                                       hipdnnBackendAttributeName_t /*attributeName*/,
                                       hipdnnBackendAttributeType_t /*attributeType*/,
                                       int64_t /*requestedElementCount*/,
                                       int64_t* /*elementCount*/,
                                       void* /*arrayOfElements*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t backendSetAttribute(hipdnnBackendDescriptor_t /*descriptor*/,
                                       hipdnnBackendAttributeName_t /*attributeName*/,
                                       hipdnnBackendAttributeType_t /*attributeType*/,
                                       int64_t /*elementCount*/,
                                       const void* /*arrayOfElements*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    const char* getErrorString(hipdnnStatus_t /*status*/) override
    {
        return "";
    }

    void getLastErrorString(char* /*message*/, size_t /*maxSize*/) override {}

    hipdnnStatus_t backendCreateAndDeserializeGraphExt(hipdnnBackendDescriptor_t* /*descriptor*/,
                                                       const uint8_t* /*serializedGraph*/,
                                                       size_t /*graphByteSize*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t versionExt(const char** /*version*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    void loggingCallbackExt(hipdnnSeverity_t /*severity*/, const char* /*msg*/) override {}

    hipdnnStatus_t setEnginePluginPathsExt(size_t /*numPaths*/,
                                           const char* const* /*pluginPaths*/,
                                           hipdnnPluginLoadingMode_ext_t /*mode*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }
};
}
