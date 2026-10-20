// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Declares identifiers matching the NVIDIA cuDNN C API, for source compatibility.
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   samples/cpp/utils/helpers.h
//     Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn.h
 * @brief cuDNN C-API subset needed by the cuDNN FE v9 graph API, forwarded to hipDNN.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/cudnn_runtime_version.h>
#include <hipdnn_compatibility/cudnn/cudnn_status.h>
#include <hipdnn_compatibility/cudnn/detail/logging_bridge.h>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>

// ===========================================================================
// C-API types
// ===========================================================================

/// @brief cuDNN handle; alias of the hipDNN handle type.
using cudnnHandle_t = ::hipdnnHandle_t;

static_assert(std::is_same_v<cudnnHandle_t, ::hipdnnHandle_t>,
              "cudnnHandle_t must alias the hipDNN handle type");

/// @brief hipDNN placeholder so signatures naming it compile; graph capture is unsupported.
using cudaGraph_t = void*;

// Other cuDNN C-API enums are omitted: the v9 graph API uses the FE-namespace
// enums aliased in cudnn_frontend_utils.h.

// Needs cudnnStatus_t; used by the entry points below.
#include <hipdnn_compatibility/cudnn/detail/status_translation.h>

// ===========================================================================
// C entry points — forward to the hipDNN backend
// ===========================================================================

extern "C" {

/// @brief Creates a hipDNN handle; also maps cuDNN FE logging env vars to hipDNN logging.
inline cudnnStatus_t cudnnCreate(cudnnHandle_t* handle)
{
    hipdnn_frontend::compatibility::cudnn_frontend::detail::configureHipdnnLoggingFromCudnnEnv();
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::toCudnnStatus(
        hipdnn_frontend::detail::hipdnnBackend()->create(handle));
}

/// @brief Destroys a handle.
inline cudnnStatus_t cudnnDestroy(cudnnHandle_t handle)
{
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::toCudnnStatus(
        hipdnn_frontend::detail::hipdnnBackend()->destroy(handle));
}

/// @brief Binds a HIP stream to a handle.
inline cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, hipStream_t streamId)
{
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::toCudnnStatus(
        hipdnn_frontend::detail::hipdnnBackend()->setStream(handle, streamId));
}

/// @brief Returns the HIP stream bound to a handle.
inline cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, hipStream_t* streamId)
{
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::toCudnnStatus(
        hipdnn_frontend::detail::hipdnnBackend()->getStream(handle, streamId));
}

/// @brief Returns the hipDNN error string for the equivalent hipDNN status.
inline const char* cudnnGetErrorString(cudnnStatus_t status)
{
    return hipdnn_frontend::detail::hipdnnBackend()->getErrorString(
        hipdnn_frontend::compatibility::cudnn_frontend::detail::toHipdnnStatus(status));
}

/// @brief Returns the claimed cuDNN runtime version, `CUDNN_VERSION`.
inline size_t cudnnGetVersion(void)
{
    return CUDNN_VERSION;
}

/// @brief Returns the claimed CUDA runtime version, `CUDNN_CUDART_VERSION`.
inline size_t cudnnGetCudartVersion(void)
{
    return CUDNN_CUDART_VERSION;
}

} // extern "C"

/// @brief Deleter that destroys and frees a heap-allocated handle; a failed destroy is logged.
struct CudnnHandleDeleter
{
    void operator()(cudnnHandle_t* handle) const
    {
        if(handle != nullptr)
        {
            // Teardown failure is not recoverable; log it and still free the allocation.
            const cudnnStatus_t status = cudnnDestroy(*handle);
            if(status != CUDNN_STATUS_SUCCESS)
            {
                HIPDNN_FE_LOG_WARN("create_cudnn_handle: cudnnDestroy failed with status "
                                   + std::to_string(static_cast<int>(status)));
            }
            delete handle;
        }
    }
};

/// @brief Creates a managed handle; returns null on failure. Mirrors the upstream sample helper.
inline std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter>
    create_cudnn_handle() // NOLINT(readability-identifier-naming)
{
    auto handle = std::make_unique<cudnnHandle_t>();
    const cudnnStatus_t status = cudnnCreate(handle.get());
    if(status != CUDNN_STATUS_SUCCESS)
    {
        HIPDNN_FE_LOG_ERROR("create_cudnn_handle: cudnnCreate failed with status "
                            + std::to_string(static_cast<int>(status)));
        return {nullptr, CudnnHandleDeleter()};
    }
    return {handle.release(), CudnnHandleDeleter()};
}
