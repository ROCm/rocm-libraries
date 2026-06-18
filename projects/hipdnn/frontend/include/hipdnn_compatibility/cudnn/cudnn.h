// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN, used under the MIT license. The C-API
// type names, enumerator names, and entry-point signatures below intentionally
// mirror NVIDIA's `cudnn.h` so that hipified v9 cuDNN consumers compile against
// hipDNN unchanged.

/**
 * @file cudnn.h
 * @brief Stub C-API header for the hipDNN cuDNN-compatibility shim (v9-only).
 *
 * This is a hand-curated, **v9-only** replacement for NVIDIA's `<cudnn.h>`
 * (RFC 0012 §4.7). It declares only the small set of C-API types that the
 * cuDNN frontend v9 graph API refers to in its own method signatures, plus the
 * handful of C entry points needed for handle init, stream binding, error
 * handling, and version checks (and to build the mirrored samples, §8.3).
 *
 * Everything here forwards to an existing hipDNN equivalent; the full cuDNN C
 * library (convolution-descriptor APIs, etc.) is intentionally out of scope.
 *
 * @note Forwarding goes through `hipdnn_frontend::detail::hipdnnBackend()` — the
 *       same mockable indirection `hipdnn_frontend/Handle.hpp` uses — so these
 *       entry points are unit-testable with the in-tree mock backend.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/cudnn_runtime_version.h>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>

// ===========================================================================
// C-API types (RFC 0012 §4.7)
// ===========================================================================

/// @brief cuDNN handle, aliased directly to the hipDNN handle type.
///
/// `hipdnnHandle_t` is the global C typedef from `<hipdnn_backend.h>`
/// (`typedef struct hipdnnHandle* hipdnnHandle_t;`), not a member of namespace
/// `hipdnn_frontend`.
using cudnnHandle_t = ::hipdnnHandle_t;

static_assert(std::is_same_v<cudnnHandle_t, ::hipdnnHandle_t>,
              "cudnnHandle_t must alias the hipDNN handle type (RFC 0012 §4.7)");

/// @brief cuDNN library status/return codes (mirrors NVIDIA `cudnnStatus_t`).
typedef enum
{
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
    CUDNN_STATUS_LICENSE_ERROR = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13,
    CUDNN_STATUS_VERSION_MISMATCH = 14,
} cudnnStatus_t;

// NOTE: This stub intentionally declares only the C-API types the *implemented*
// entry points use — `cudnnHandle_t` and `cudnnStatus_t`. The remaining v9
// C-API enums named in RFC 0012 §4.7 (`cudnnDataType_t`, `cudnnTensorFormat_t`,
// `cudnnConvolutionMode_t`, `cudnnReduceTensorOp_t`, `cudnnNormFwdPhase_t`,
// `cudnnBackendHeurMode_t`, `cudnnBackendNumericalNote_t`,
// `cudnnBackendBehaviorNote_t`, `cudnnBackendDescriptorType_t`) land with the
// type-mapping work, where their values/aliasing are verified against upstream
// rather than stubbed here.

// Status translation between the cuDNN and hipDNN enum families. Included after
// the C-API types above (it needs cudnnStatus_t) and before the entry points
// below (which use it). Lives under detail/ and is shim-internal.
#include <hipdnn_compatibility/cudnn/detail/status_translation.h>

// ===========================================================================
// C entry points (RFC 0012 §4.7) — forward to the hipDNN backend
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
// Mirrors the helper NVIDIA ships in its sample utilities
// (samples/cpp/utils/helpers.h), so mirrored sample code that calls
// `create_cudnn_handle()` works unchanged (RFC 0012 §4.7, §8.3). Unlike the
// upstream helper, this version does not depend on a test framework.

/// @brief RAII deleter that destroys a heap-allocated cuDNN handle.
struct CudnnHandleDeleter
{
    void operator()(cudnnHandle_t* handle) const
    {
        if(handle != nullptr)
        {
            cudnnDestroy(*handle);
            delete handle;
        }
    }
};

/// @brief Create a managed cuDNN handle (mirrors NVIDIA's sample helper).
///
/// The snake_case name intentionally mirrors NVIDIA's helper so mirrored sample
/// code compiles unchanged; the naming check is suppressed accordingly.
inline std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter>
    create_cudnn_handle() // NOLINT(readability-identifier-naming)
{
    auto handle = std::make_unique<cudnnHandle_t>();
    cudnnCreate(handle.get());
    return {handle.release(), CudnnHandleDeleter()};
}
