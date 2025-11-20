// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <cstdint>
#include <gmock/gmock.h>

#define HIPDNN_BACKEND_STATIC_DEFINE

#include "hipdnn_backend.h"

// NOLINTBEGIN

class Mock_hipdnn_backend : public hipdnn_frontend::IHipdnnBackend
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
    MOCK_METHOD(hipdnnStatus_t,
                backendCreateAndDeserializeGraphExt,
                (hipdnnBackendDescriptor_t * descriptor,
                 const uint8_t* serialized_graph,
                 size_t graph_byte_size),
                ());
    MOCK_METHOD(void, loggingCallbackExt, (hipdnnSeverity_t severity, const char* msg), ());
    MOCK_METHOD(hipdnnStatus_t,
                setEnginePluginPathsExt,
                (size_t num_paths,
                 const char* const* plugin_paths,
                 hipdnnPluginLoadingMode_ext_t mode),
                ());
};

// NOLINTEND
