// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN, used under the MIT license.

/**
 * @file cudnn.h
 * @brief Stub C-API header for the hipDNN cuDNN-compatibility shim (v9-only).
 *
 * Hand-curated, v9-only replacement for NVIDIA's `<cudnn.h>`: declares the few
 * C-API types the cuDNN frontend v9 graph API names in its signatures, plus the
 * C entry points for handle init, stream binding, error handling, and version
 * checks. Everything forwards to an existing hipDNN equivalent.
 *
 * @note Forwarding goes through `hipdnn_frontend::detail::hipdnnBackend()` — the
 *       same mockable indirection `Handle.hpp` uses — so these entry points are
 *       unit-testable with the in-tree mock backend.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/cudnn_runtime_version.h>
#include <hipdnn_compatibility/cudnn/cudnn_status.h>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>

// ===========================================================================
// C-API types
// ===========================================================================

/// @brief cuDNN handle, aliased directly to the hipDNN handle type.
///
/// `hipdnnHandle_t` is the global C typedef from `<hipdnn_backend.h>`, not a
/// member of namespace `hipdnn_frontend`.
using cudnnHandle_t = ::hipdnnHandle_t;

static_assert(std::is_same_v<cudnnHandle_t, ::hipdnnHandle_t>,
              "cudnnHandle_t must alias the hipDNN handle type");

// Only the C-API types the v9 graph API actually references are declared here
// (cudnnHandle_t, plus cudnnStatus_t from cudnn_status.h). Other cuDNN C-API
// enums are intentionally omitted: the v9 graph surface uses the FE-namespace
// enums (DataType_t, …) aliased in cudnn_frontend_utils.h, not the C-API ones.

// Status translation. Included after cudnnStatus_t above and before the entry
// points below, which use it.
#include <hipdnn_compatibility/cudnn/detail/status_translation.h>

// ===========================================================================
// C entry points — forward to the hipDNN backend
// ===========================================================================

extern "C" {

/// @brief Create a cuDNN/hipDNN handle. Mirrors NVIDIA `cudnnCreate`.
inline cudnnStatus_t cudnnCreate(cudnnHandle_t* handle)
{
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::toCudnnStatus(
        hipdnn_frontend::detail::hipdnnBackend()->create(handle));
}

/// @brief Destroy a handle. Mirrors NVIDIA `cudnnDestroy`.
inline cudnnStatus_t cudnnDestroy(cudnnHandle_t handle)
{
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::toCudnnStatus(
        hipdnn_frontend::detail::hipdnnBackend()->destroy(handle));
}

/// @brief Bind a stream to a handle. Mirrors NVIDIA `cudnnSetStream`.
inline cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, hipStream_t streamId)
{
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::toCudnnStatus(
        hipdnn_frontend::detail::hipdnnBackend()->setStream(handle, streamId));
}

/// @brief Query the stream bound to a handle. Mirrors NVIDIA `cudnnGetStream`.
inline cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, hipStream_t* streamId)
{
    return hipdnn_frontend::compatibility::cudnn_frontend::detail::toCudnnStatus(
        hipdnn_frontend::detail::hipdnnBackend()->getStream(handle, streamId));
}

/// @brief Return a human-readable string for a status. Mirrors NVIDIA `cudnnGetErrorString`.
inline const char* cudnnGetErrorString(cudnnStatus_t status)
{
    return hipdnn_frontend::detail::hipdnnBackend()->getErrorString(
        hipdnn_frontend::compatibility::cudnn_frontend::detail::toHipdnnStatus(status));
}

/// @brief Return the claimed cuDNN runtime version. Mirrors NVIDIA `cudnnGetVersion`.
inline size_t cudnnGetVersion(void)
{
    return CUDNN_VERSION;
}

} // extern "C"

// ===========================================================================
// create_cudnn_handle() convenience helper
// ===========================================================================
// Mirrors the helper in NVIDIA's sample utilities so mirrored sample code
// compiles unchanged, minus the upstream test-framework dependency.

/// @brief RAII deleter that destroys a heap-allocated cuDNN handle.
struct CudnnHandleDeleter
{
    void operator()(cudnnHandle_t* handle) const
    {
        if(handle != nullptr)
        {
            // A failed destroy at teardown is not recoverable; log and ignore
            // (the heap allocation is still freed below).
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

/// @brief Create a managed cuDNN handle (mirrors NVIDIA's sample helper).
///
/// snake_case name mirrors NVIDIA's so sample code compiles unchanged. On a
/// backend create failure, logs and returns an empty (null) pointer.
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
