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
 * @brief cuDNN FE `detail::` wrappers over the emulated cuDNN C API, plus the plan timer.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include <hipdnn_compatibility/cudnn/cudnn.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

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

// Replaces upstream's CUDA event wrappers: device time between start() and stop() on the
// handle's stream, measured by one hipDNN profiling descriptor reused across measurements.
class PlanTimer
{
public:
    explicit PlanTimer(cudnnHandle_t handle)
        : _handle(handle)
        , _profiling(HIPDNN_BACKEND_PROFILING_CONTROL_EXT)
    {
    }

    cudnnStatus_t start()
    {
        if(!_profiling.valid())
        {
            return CUDNN_STATUS_NOT_INITIALIZED;
        }
        cudnnStatus_t status = setTrigger(HIPDNN_ATTR_PROFILING_RESET_EXT);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            return status;
        }
        status = toCudnnStatus(hipdnn_frontend::detail::hipdnnBackend()->backendSetAttribute(
            _profiling.get(),
            HIPDNN_ATTR_PROFILING_HANDLE_EXT,
            HIPDNN_TYPE_HANDLE,
            1,
            static_cast<const void*>(&_handle)));
        if(status != CUDNN_STATUS_SUCCESS)
        {
            return status;
        }
        return setTrigger(HIPDNN_ATTR_PROFILING_START_EXT);
    }

    cudnnStatus_t stop()
    {
        if(!_profiling.valid())
        {
            return CUDNN_STATUS_NOT_INITIALIZED;
        }
        const cudnnStatus_t status = setTrigger(HIPDNN_ATTR_PROFILING_STOP_EXT);
        if(status != CUDNN_STATUS_SUCCESS)
        {
            return status;
        }
        return toCudnnStatus(
            hipdnn_frontend::detail::hipdnnBackend()->backendFinalize(_profiling.get()));
    }

    cudnnStatus_t elapsedMs(float* ms) const
    {
        if(!_profiling.valid())
        {
            return CUDNN_STATUS_NOT_INITIALIZED;
        }
        if(ms == nullptr)
        {
            return CUDNN_STATUS_BAD_PARAM;
        }
        float value = 0.0F;
        const cudnnStatus_t status
            = toCudnnStatus(hipdnn_frontend::detail::hipdnnBackend()->backendGetAttribute(
                _profiling.get(),
                HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT,
                HIPDNN_TYPE_FLOAT,
                1,
                nullptr,
                &value));
        if(status != CUDNN_STATUS_SUCCESS)
        {
            return status;
        }
        // The backend passes a finite negative span through as an invalid measurement.
        if(!std::isfinite(value) || value < 0.0F)
        {
            return CUDNN_STATUS_EXECUTION_FAILED;
        }
        *ms = value;
        return CUDNN_STATUS_SUCCESS;
    }

private:
    cudnnStatus_t setTrigger(hipdnnBackendAttributeName_t attribute)
    {
        const bool trigger = true;
        return toCudnnStatus(hipdnn_frontend::detail::hipdnnBackend()->backendSetAttribute(
            _profiling.get(), attribute, HIPDNN_TYPE_BOOLEAN, 1, &trigger));
    }

    cudnnHandle_t _handle;
    hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor _profiling;
};

// Upstream's cuda_device_synchronize; the backend synchronizes on SET, so there is no finalize.
inline cudnnStatus_t deviceSynchronize()
{
    const hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor sync(
        HIPDNN_BACKEND_PROFILING_CONTROL_EXT);
    if(!sync.valid())
    {
        return CUDNN_STATUS_NOT_INITIALIZED;
    }
    const bool trigger = true;
    return toCudnnStatus(hipdnn_frontend::detail::hipdnnBackend()->backendSetAttribute(
        sync.get(), HIPDNN_ATTR_PROFILING_DEVICE_SYNC_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trigger));
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
