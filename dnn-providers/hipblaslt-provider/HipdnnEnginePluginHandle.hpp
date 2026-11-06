// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hip/hip_runtime_api.h>
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#endif
#include <memory>
#include <unordered_map>
#if defined(HIPBLASLTPROVIDER_ENABLE_ROCBLAS) || defined(HIPBLASLTPROVIDER_ENABLE_CONV)
#include <mutex>
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
#include <rocblas/rocblas.h>
#endif

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "HipblasltContainer.hpp"
#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
#include "HipblasltUtils.hpp"
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_CONV
#include "engines/conv/RepackProgram.hpp"
#endif

// NOLINTBEGIN
struct HipdnnEnginePluginHandle
{
public:
    virtual ~HipdnnEnginePluginHandle()
    {
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
        if(_rocblasHandle != nullptr)
        {
            rocblas_destroy_handle(_rocblasHandle);
        }
#endif
    }

#ifdef HIPBLASLTPROVIDER_ENABLE_HIPBLASLT
    hipblasLtHandle_t hipblasltHandle = nullptr;
#endif

#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    /// The rocBLAS handle, created on first use: creating one loads rocBLAS's kernel library,
    /// a cost a handle that only ever serves hipBLASLt graphs should not pay.
    rocblas_handle rocblasHandle() const
    {
        std::call_once(_rocblasOnce, [this] {
            if(rocblas_create_handle(&_rocblasHandle) != rocblas_status_success)
            {
                _rocblasHandle = nullptr;
            }
        });
        if(_rocblasHandle == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                           "Failed to create rocBLAS handle");
        }
        return _rocblasHandle;
    }
#endif

#ifdef HIPBLASLTPROVIDER_ENABLE_CONV
    /// The convolution repack kernels, compiled for the current device on first use: only a
    /// convolution that repacks pays for the compile.
    const hipblaslt_plugin::RepackProgram& repackProgram() const
    {
        std::call_once(_repackOnce, [this] {
            _repackProgram = std::make_unique<hipblaslt_plugin::RepackProgram>();
        });
        return *_repackProgram;
    }
#endif

    void setStream(hipStream_t stream)
    {
        _stream = stream;
    }

    hipStream_t getStream() const
    {
        return _stream;
    }

    std::shared_ptr<hipblaslt_plugin::HipblasltContainer> hipblasltContainer;
    hipblaslt_plugin::EngineManager& getEngineManager()
    {
        return hipblasltContainer->getEngineManager();
    }

    void storeEngineDetailsDetachedBuffer(const void* ptr,
                                          std::unique_ptr<flatbuffers::DetachedBuffer> buffer)
    {
        HIPDNN_PLUGIN_LOG_INFO("Storing detached buffer at address: " << ptr);
        _engineDetailsBuffers[ptr] = std::move(buffer);
    }

    void removeEngineDetailsDetachedBuffer(const void* ptr)
    {
        HIPDNN_PLUGIN_LOG_INFO("Removing detached buffer at address: " << ptr);

        auto it = _engineDetailsBuffers.find(ptr);
        if(it != _engineDetailsBuffers.end())
        {
            _engineDetailsBuffers.erase(it);
        }
        else
        {
            HIPDNN_PLUGIN_LOG_WARN(
                "No detached buffer found at address: "
                << ptr
                << ". Could not remove engine "
                   "details. Ensure you "
                   "are using the same hipdnn handle you used for engine details creation");
        }
    }

private:
#ifdef HIPBLASLTPROVIDER_ENABLE_ROCBLAS
    mutable std::once_flag _rocblasOnce;
    mutable rocblas_handle _rocblasHandle = nullptr;
#endif
#ifdef HIPBLASLTPROVIDER_ENABLE_CONV
    mutable std::once_flag _repackOnce;
    mutable std::unique_ptr<hipblaslt_plugin::RepackProgram> _repackProgram;
#endif
    hipStream_t _stream = nullptr;
    std::unordered_map<const void*, std::unique_ptr<flatbuffers::DetachedBuffer>>
        _engineDetailsBuffers;
};

// NOLINTEND
