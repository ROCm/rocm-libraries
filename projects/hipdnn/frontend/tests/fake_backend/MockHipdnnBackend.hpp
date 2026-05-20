// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <cstdint>
#include <gmock/gmock.h>

#define HIPDNN_BACKEND_STATIC_DEFINE

#include "hipdnn_backend.h"
#include <hipdnn_frontend/detail/HipdnnBackendInterface.hpp>

// NOLINTBEGIN

class Mock_hipdnn_backend : public hipdnn_frontend::detail::IHipdnnBackend
{
public:
    MOCK_METHOD(hipdnnStatus_t, create, (hipdnnHandle_t * handle), ());
    MOCK_METHOD(hipdnnStatus_t, destroy, (hipdnnHandle_t handle), ());
    MOCK_METHOD(hipdnnStatus_t, setStream, (hipdnnHandle_t handle, hipStream_t streamId), ());
    MOCK_METHOD(hipdnnStatus_t, getStream, (hipdnnHandle_t handle, hipStream_t* streamId), ());
    MOCK_METHOD(hipdnnStatus_t,
                backendCreateDescriptor,
                (hipdnnBackendDescriptorType_t descriptor_type,
                 hipdnnBackendDescriptor_t* descriptor),
                ());
    MOCK_METHOD(hipdnnStatus_t,
                backendDestroyDescriptor,
                (hipdnnBackendDescriptor_t descriptor),
                ());
    MOCK_METHOD(hipdnnStatus_t,
                backendExecute,
                (hipdnnHandle_t handle,
                 hipdnnBackendDescriptor_t execution_plan,
                 hipdnnBackendDescriptor_t variant_pack),
                ());
    MOCK_METHOD(hipdnnStatus_t, backendFinalize, (hipdnnBackendDescriptor_t descriptor), ());
    MOCK_METHOD(hipdnnStatus_t,
                backendGetAttribute,
                (hipdnnBackendDescriptor_t descriptor,
                 hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t requested_element_count,
                 int64_t* element_count,
                 void* array_of_elements),
                ());
    MOCK_METHOD(hipdnnStatus_t,
                backendSetAttribute,
                (hipdnnBackendDescriptor_t descriptor,
                 hipdnnBackendAttributeName_t attribute_name,
                 hipdnnBackendAttributeType_t attribute_type,
                 int64_t element_count,
                 const void* array_of_elements),
                ());
    MOCK_METHOD(const char*, getErrorString, (hipdnnStatus_t status), ());
    MOCK_METHOD(void, getLastErrorString, (char* message, size_t max_size), ());
    MOCK_METHOD(hipdnn_data_sdk::utilities::Version, version, ());
    MOCK_METHOD(const char*, versionString, ());
    MOCK_METHOD(hipdnnStatus_t,
                backendCreateAndDeserializeGraphExt,
                (hipdnnBackendDescriptor_t * descriptor,
                 const uint8_t* serialized_graph,
                 size_t graph_byte_size),
                ());
    MOCK_METHOD(hipdnnStatus_t,
                backendGetSerializedBinaryGraphExt,
                (hipdnnBackendDescriptor_t descriptor,
                 size_t requestedByteSize,
                 size_t* graphByteSize,
                 uint8_t* serializedGraph),
                (override));
    MOCK_METHOD(hipdnnStatus_t,
                backendGetSerializedJsonGraphExt,
                (hipdnnBackendDescriptor_t descriptor,
                 size_t requestedByteSize,
                 size_t* graphByteSize,
                 char* serializedJsonGraph),
                (override));
    MOCK_METHOD(hipdnnStatus_t,
                backendCreateAndDeserializeJsonGraphExt,
                (hipdnnBackendDescriptor_t * descriptor,
                 const char* jsonGraph,
                 size_t jsonByteSize),
                (override));
    MOCK_METHOD(hipdnnStatus_t,
                backendGetSerializedExecutionPlanExt,
                (hipdnnBackendDescriptor_t descriptor,
                 size_t requestedByteSize,
                 size_t* planByteSize,
                 uint8_t* serializedPlan),
                (override));
    MOCK_METHOD(hipdnnStatus_t,
                backendCreateAndDeserializeExecutionPlanExt,
                (hipdnnHandle_t handle,
                 hipdnnBackendDescriptor_t* descriptor,
                 const uint8_t* serializedPlan,
                 size_t planByteSize),
                (override));
    MOCK_METHOD(void, loggingCallbackExt, (hipdnnSeverity_t severity, const char* msg), ());
    MOCK_METHOD(hipdnnStatus_t,
                setEnginePluginPathsExt,
                (size_t num_paths,
                 const char* const* plugin_paths,
                 hipdnnPluginLoadingMode_ext_t mode),
                ());
    MOCK_METHOD(
        hipdnnStatus_t,
        getLoadedEnginePluginPathsExt,
        (hipdnnHandle_t handle, size_t* numPluginPaths, char** pluginPaths, size_t* maxStringLen),
        ());
};

// NOLINTEND
