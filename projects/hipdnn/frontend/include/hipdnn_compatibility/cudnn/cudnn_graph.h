// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Declares identifiers matching the NVIDIA cuDNN C API, for source compatibility.
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_shim.h
//     Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_graph.h
 * @brief cuDNN backend descriptor C API for the hipDNN cuDNN-compatibility shim.
 *
 * Header-only replacement for the backend-descriptor half of NVIDIA's
 * `<cudnn_graph.h>`. Descriptors live in the shim; hipDNN is reached only when a
 * graph is lowered. Includable standalone.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <hipdnn_compatibility/cudnn/cudnn_status.h>
#include <hipdnn_compatibility/cudnn/detail/backend/backend_api.h>
#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>

extern "C" {

/// @brief Create a backend descriptor. Mirrors NVIDIA `cudnnBackendCreateDescriptor`.
inline cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t descriptorType,
                                                  cudnnBackendDescriptor_t* descriptor)
{
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::backend::createDescriptor(
        descriptorType, descriptor);
}

/// @brief Destroy a backend descriptor. Mirrors NVIDIA `cudnnBackendDestroyDescriptor`.
inline cudnnStatus_t cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t descriptor)
{
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::backend::destroyDescriptor(
        descriptor);
}

/// @brief Set a descriptor attribute. Mirrors NVIDIA `cudnnBackendSetAttribute`.
inline cudnnStatus_t cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor,
                                              cudnnBackendAttributeName_t attributeName,
                                              cudnnBackendAttributeType_t attributeType,
                                              int64_t elementCount,
                                              const void* arrayOfElements)
{
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::backend::setAttribute(
        descriptor, attributeName, attributeType, elementCount, arrayOfElements);
}

/// @brief Get a descriptor attribute. Mirrors NVIDIA `cudnnBackendGetAttribute`.
inline cudnnStatus_t cudnnBackendGetAttribute(cudnnBackendDescriptor_t descriptor,
                                              cudnnBackendAttributeName_t attributeName,
                                              cudnnBackendAttributeType_t attributeType,
                                              int64_t requestedElementCount,
                                              int64_t* elementCount,
                                              void* arrayOfElements)
{
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::backend::getAttribute(
        descriptor,
        attributeName,
        attributeType,
        requestedElementCount,
        elementCount,
        arrayOfElements);
}

/// @brief Finalize a backend descriptor. Mirrors NVIDIA `cudnnBackendFinalize`.
inline cudnnStatus_t cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor)
{
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::backend::finalize(descriptor);
}

/// @brief Execute an execution plan. Mirrors NVIDIA `cudnnBackendExecute`.
inline cudnnStatus_t cudnnBackendExecute(cudnnHandle_t handle,
                                         cudnnBackendDescriptor_t executionPlan,
                                         cudnnBackendDescriptor_t variantPack)
{
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::backend::execute(
        handle, executionPlan, variantPack);
}

/// @brief Copy and clear this thread's last error. Mirrors NVIDIA `cudnnGetLastErrorString`.
inline void cudnnGetLastErrorString(char* message, size_t maxSize)
{
    hipdnn_frontend::compatibility::cudnn_frontend::detail::backend::getLastErrorString(message,
                                                                                        maxSize);
}

} // extern "C"
