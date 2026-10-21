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
 * @brief cuDNN backend C types and plan execution for the hipDNN cuDNN-compatibility shim.
 *
 * Header-only replacement for the parts of NVIDIA's `<cudnn_graph.h>` the shim keeps:
 * the backend C types, `cudnnBackendExecute` and `cudnnGetLastErrorString`, both of
 * which pass straight through to hipDNN. Includable standalone.
 */

#pragma once

#include <cstddef>

#include <hipdnn_compatibility/cudnn/cudnn_status.h>
#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/status_translation.h>
#include <hipdnn_compatibility/cudnn/detail/v8/boundary.h>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>

extern "C" {

/// @brief Execute an execution plan. Mirrors NVIDIA `cudnnBackendExecute`.
inline cudnnStatus_t cudnnBackendExecute(cudnnHandle_t handle,
                                         cudnnBackendDescriptor_t executionPlan,
                                         cudnnBackendDescriptor_t variantPack)
{
    namespace shim = hipdnn_frontend::compatibility::cudnn_frontend::detail;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    const shim::v8::Outcome outcome = shim::v8::guarded([&] {
        status = shim::toCudnnStatus(hipdnn_frontend::detail::hipdnnBackend()->backendExecute(
            handle, executionPlan, variantPack));
    });
    return outcome ? status : outcome.status;
}

/// @brief Copy and clear this thread's last hipDNN error. Mirrors NVIDIA `cudnnGetLastErrorString`.
inline void cudnnGetLastErrorString(char* message, size_t maxSize)
{
    if(message == nullptr || maxSize == 0)
    {
        return;
    }
    message[0] = '\0';
    const auto outcome = hipdnn_frontend::compatibility::cudnn_frontend::detail::v8::guarded(
        [&] { hipdnn_frontend::detail::hipdnnBackend()->getLastErrorString(message, maxSize); });
    if(!outcome)
    {
        message[0] = '\0';
    }
}

} // extern "C"
