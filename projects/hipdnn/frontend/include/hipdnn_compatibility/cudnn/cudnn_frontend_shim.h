// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_shim.h
//     Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_shim.h
 * @brief cuDNN FE `detail::` wrappers over the emulated cuDNN C API (descriptor subset).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include <hipdnn_compatibility/cudnn/cudnn.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>

// The shim never loads cuDNN dynamically, so the runtime version checks are empty.
#define NV_CUDNN_FE_DYNAMIC_CHECK_BACKEND_DESCRIPTOR(MINIMUM_VERSION, DESCRIPTOR, MESSAGE)
#define NV_CUDNN_FE_DYNAMIC_CHECK_CUDNN_BACKEND_VERSION(MINIMUM_VERSION, STATUS)

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_shim.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-redundant-void-arg,readability-redundant-inline-specifier,misc-misplaced-const,readability-avoid-return-with-void-value,misc-const-correctness,bugprone-unused-local-non-trivial-variable):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

inline cudnnStatus_t create_handle(cudnnHandle_t* handle)
{
    return cudnnCreate(handle);
}

inline cudnnStatus_t destroy_handle(cudnnHandle_t handle)
{
    return cudnnDestroy(handle);
}

inline constexpr size_t get_compiled_version(void)
{
    return CUDNN_VERSION;
}

inline std::string convert_version_to_str(const size_t version)
{
    // The multiplier for major version pre-v9 and post-v9 are different.
    size_t major = version / 10000;
    size_t minor = (version / 100) % 100;
    if(major == 0)
    {
        major = version / 1000;
        minor = (version / 100) % 10;
    }
    auto patch = version % 100;

    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

inline std::string get_backend_version_string()
{
    return convert_version_to_str(get_backend_version());
}

inline cudnnStatus_t create_descriptor(cudnnBackendDescriptorType_t descriptorType,
                                       cudnnBackendDescriptor_t* descriptor)
{
    return cudnnBackendCreateDescriptor(descriptorType, descriptor);
}

inline cudnnStatus_t destroy_descriptor(cudnnBackendDescriptor_t descriptor)
{
    return cudnnBackendDestroyDescriptor(descriptor);
}

inline cudnnStatus_t set_attribute(cudnnBackendDescriptor_t descriptor,
                                   cudnnBackendAttributeName_t attributeName,
                                   cudnnBackendAttributeType_t attributeType,
                                   int64_t elementCount,
                                   const void* arrayOfElements)
{
    return cudnnBackendSetAttribute(
        descriptor, attributeName, attributeType, elementCount, arrayOfElements);
}

inline cudnnStatus_t get_attribute(const cudnnBackendDescriptor_t descriptor,
                                   cudnnBackendAttributeName_t attributeName,
                                   cudnnBackendAttributeType_t attributeType,
                                   int64_t requestedElementCount,
                                   int64_t* elementCount,
                                   void* arrayOfElements)
{
    return cudnnBackendGetAttribute(descriptor,
                                    attributeName,
                                    attributeType,
                                    requestedElementCount,
                                    elementCount,
                                    arrayOfElements);
}

inline cudnnStatus_t finalize(cudnnBackendDescriptor_t descriptor)
{
    return cudnnBackendFinalize(descriptor);
}

inline cudnnStatus_t execute(cudnnHandle_t handle,
                             cudnnBackendDescriptor_t executionPlan,
                             cudnnBackendDescriptor_t variantPack)
{
    return cudnnBackendExecute(handle, executionPlan, variantPack);
}

inline const char* get_error_string(cudnnStatus_t status)
{
    return cudnnGetErrorString(status);
}

inline void get_last_error_string(char* message, size_t size)
{
    if(detail::get_backend_version() >= 90000 && detail::get_compiled_version() >= 90000)
    {
#if CUDNN_VERSION >= 90000
        return cudnnGetLastErrorString(message, size);
#endif
    }
    else
    {
        std::string default_message
            = "Can't retrieve backend error messages for CUDNN version < 9.0";
        // strncpy(message, default_message.c_str(), size - 1);
        message[size - 1] = '\0'; // Ensure null terminator at the end of the string
    }
}

inline std::string get_last_error_string_()
{
    const size_t size = 65535;

    std::string message;

    message.resize(size);

    get_last_error_string(message.data(), size);

    message.resize(std::strlen(message.c_str()));

    return message;
}

inline cudnnStatus_t set_stream(cudnnHandle_t handle, hipStream_t stream)
{
    return cudnnSetStream(handle, stream);
}

inline cudnnStatus_t get_stream(cudnnHandle_t handle, hipStream_t* stream)
{
    return cudnnGetStream(handle, stream);
}

// NOLINTEND(readability-identifier-naming,modernize-redundant-void-arg,readability-redundant-inline-specifier,misc-misplaced-const,readability-avoid-return-with-void-value,misc-const-correctness,bugprone-unused-local-non-trivial-variable)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
