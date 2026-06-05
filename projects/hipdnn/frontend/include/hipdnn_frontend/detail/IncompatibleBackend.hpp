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

    hipdnnStatus_t backendGetSerializedBinaryGraphExt(hipdnnBackendDescriptor_t /*descriptor*/,
                                                      size_t /*requestedByteSize*/,
                                                      size_t* /*graphByteSize*/,
                                                      uint8_t* /*serializedGraph*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t backendGetSerializedJsonGraphExt(hipdnnBackendDescriptor_t /*descriptor*/,
                                                    size_t /*requestedByteSize*/,
                                                    size_t* /*graphByteSize*/,
                                                    char* /*serializedJsonGraph*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t
        backendCreateAndDeserializeJsonGraphExt(hipdnnBackendDescriptor_t* /*descriptor*/,
                                                const char* /*jsonGraph*/,
                                                size_t /*jsonByteSize*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t backendGetSerializedExecutionPlanExt(hipdnnBackendDescriptor_t /*descriptor*/,
                                                        size_t /*requestedByteSize*/,
                                                        size_t* /*planByteSize*/,
                                                        uint8_t* /*serializedPlan*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t
        backendCreateAndDeserializeExecutionPlanExt(hipdnnHandle_t /*handle*/,
                                                    hipdnnBackendDescriptor_t* /*descriptor*/,
                                                    const uint8_t* /*serializedPlan*/,
                                                    size_t /*planByteSize*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnn_data_sdk::utilities::Version version() override
    {
        return hipdnn_data_sdk::utilities::Version{-1, 0, 0};
    }

    const char* versionString() override
    {
        return "";
    }

    void loggingCallbackExt(hipdnnSeverity_t /*severity*/, const char* /*msg*/) override {}

    hipdnnStatus_t setEnginePluginPathsExt(size_t /*numPaths*/,
                                           const char* const* /*pluginPaths*/,
                                           hipdnnPluginLoadingMode_ext_t /*mode*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t getLoadedEnginePluginPathsExt(hipdnnHandle_t /*handle*/,
                                                 size_t* /*numPluginPaths*/,
                                                 char** /*pluginPaths*/,
                                                 size_t* /*maxStringLen*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    // RFC 0007 Section 16: Heuristic policy enumeration
    hipdnnStatus_t getHeuristicPolicyCount(hipdnnHandle_t /*handle*/,
                                           size_t* /*numPolicies*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }

    hipdnnStatus_t getHeuristicPolicyInfo(hipdnnHandle_t /*handle*/,
                                          size_t /*policyIndex*/,
                                          int64_t* /*policyId*/,
                                          char* /*policyName*/,
                                          size_t* /*policyNameLen*/,
                                          char* /*pluginName*/,
                                          size_t* /*pluginNameLen*/,
                                          char* /*pluginVersion*/,
                                          size_t* /*pluginVersionLen*/,
                                          char* /*apiVersion*/,
                                          size_t* /*apiVersionLen*/) override
    {
        return hipdnnStatus_t::HIPDNN_STATUS_NOT_INITIALIZED;
    }
};

} // namespace hipdnn_frontend::detail
